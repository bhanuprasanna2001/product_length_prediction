"""MLP regressor with multi-embedding ensemble for product length prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .config import Config
from .losses import get_loss_fn, compute_mape, compute_rmsle, compute_score, EPSILON


class MLPHead(nn.Module):
    """Multi-layer perceptron with batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EnsembleModel(pl.LightningModule):
    """Embedding ensemble → MLP regressor for product length prediction.

    Architecture:
        [text_emb₁ ‖ text_emb₂ ‖ ... ‖ type_emb ‖ knn_proj] → MLP → length

    Features:
        - Concatenates pre-computed text embeddings from multiple models
        - Learns a product-type embedding (index 0 reserved for unknown types)
        - Optionally projects KNN retrieval features through a small MLP
        - Supports log-target transform for handling skewed distributions
    """

    def __init__(self, config: Config, num_product_types: int):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.use_log_target = config.use_log_target

        # Product type embedding
        self.product_emb = nn.Embedding(
            num_product_types, config.product_type_emb_dim, padding_idx=0
        )
        nn.init.normal_(self.product_emb.weight, mean=0, std=0.02)

        # Compute input dimensionality
        input_dim = config.total_embedding_dim + config.product_type_emb_dim

        # Optional KNN feature projection
        knn_proj_dim = 32
        if config.knn_dim > 0:
            self.knn_proj = nn.Sequential(
                nn.Linear(config.knn_dim, knn_proj_dim),
                nn.ReLU(),
                nn.Linear(knn_proj_dim, knn_proj_dim),
            )
            input_dim += knn_proj_dim
        else:
            self.knn_proj = None

        self.head = MLPHead(
            input_dim, config.hidden_dims, config.dropout, config.use_batch_norm
        )
        self.loss_fn = get_loss_fn(config.loss_fn)

    def forward(
        self,
        text_embedding: torch.Tensor,
        product_type: torch.Tensor,
        knn_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [text_embedding, self.product_emb(product_type)]
        if knn_features is not None and self.knn_proj is not None:
            parts.append(self.knn_proj(knn_features))
        return self.head(torch.cat(parts, dim=-1))

    # ── Training / Validation / Test steps ─────────────────────

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        pred = self(batch["text_embedding"], batch["product_type"], batch.get("knn_features"))
        pred_safe = F.relu(pred) + EPSILON
        target = batch["target"]

        # Transform back to linear scale for loss/metrics when using log-target
        if self.use_log_target:
            pred_linear = torch.expm1(pred_safe)
            target_linear = torch.expm1(target)
        else:
            pred_linear, target_linear = pred_safe, target

        loss = self.loss_fn(pred_linear, target_linear)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mape", compute_mape(pred_linear, target_linear), prog_bar=True, on_epoch=True)
        self.log(f"{stage}_rmsle", compute_rmsle(pred_linear, target_linear), on_epoch=True)
        self.log(f"{stage}_score", compute_score(pred_linear, target_linear), on_epoch=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pred = self(batch["text_embedding"], batch["product_type"], batch.get("knn_features"))
        pred_safe = F.relu(pred) + EPSILON
        if self.use_log_target:
            pred_safe = torch.expm1(pred_safe)
        return pred_safe

    # ── Optimizer + scheduler ──────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

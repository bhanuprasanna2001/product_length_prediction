"""MLP regressor for pre-computed embedding ensembles with optional KNN features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from ..config import Config
from ..utils.losses import get_loss_fn, compute_mape, compute_rmsle


class MLPHead(nn.Module):
    """Multi-layer perceptron regressor with batch normalization and dropout."""
    
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h)])
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
    
    Combines pre-computed text embeddings with product type embeddings and
    optional KNN features, then feeds through MLP to predict length.
    Supports log-target transform for handling skewed distributions.
    """
    
    def __init__(self, config: Config, num_product_types: int, knn_dim: int = 0, use_log_target: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.knn_dim = knn_dim
        self.use_log_target = use_log_target
        
        # Product type embedding (index 0 reserved for unknown)
        self.product_emb = nn.Embedding(num_product_types, config.model.product_type_emb_dim, padding_idx=0)
        nn.init.normal_(self.product_emb.weight, mean=0, std=0.02)
        
        # Input dimension calculation
        input_dim = config.embeddings.total_dim + config.model.product_type_emb_dim
        
        # Optional KNN feature projection (32-dim learned projection)
        if knn_dim > 0:
            self.knn_proj = nn.Sequential(nn.Linear(knn_dim, 32), nn.ReLU(), nn.Linear(32, 32))
            input_dim += 32
        else:
            self.knn_proj = None
        
        self.head = MLPHead(input_dim, config.model.hidden_dims, config.model.dropout, config.model.use_batch_norm)
        self.loss_fn = get_loss_fn(config.training.loss_fn)
        
    def forward(self, text_embedding: torch.Tensor, product_type: torch.Tensor, knn_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        type_emb = self.product_emb(product_type)
        
        if knn_features is not None and self.knn_proj is not None:
            parts = [text_embedding, type_emb, self.knn_proj(knn_features)]
        else:
            parts = [text_embedding, type_emb]
            
        return self.head(torch.cat(parts, dim=-1))
    
    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pred_safe = F.relu(pred) + 1e-6
            return {"mape": compute_mape(pred_safe, target), "rmsle": compute_rmsle(pred_safe, target)}
    
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        pred = self(batch["text_embedding"], batch["product_type"], batch.get("knn_features"))
        pred_safe = F.relu(pred) + 1e-6
        loss = self.loss_fn(pred_safe, batch["target"])
        metrics = self._compute_metrics(pred, batch["target"])
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_mape", metrics["mape"], prog_bar=True, on_epoch=True)
        self.log(f"{stage}_rmsle", metrics["rmsle"], on_epoch=True)
        return loss
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")
    
    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")
    
    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        pred = self(batch["text_embedding"], batch["product_type"], batch.get("knn_features"))
        if self.use_log_target:
            pred = torch.expm1(pred)
        return F.relu(pred) + 1e-6
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(self.config.training.warmup_ratio * total_steps)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.training.lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps, anneal_strategy="cos"
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

"""
Ensemble Model Architecture
===========================
Lightweight MLP head for pre-computed embedding ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..config import Config


class MLPHead(nn.Module):
    """MLP regressor with optional batch normalization."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class EnsembleModel(pl.LightningModule):
    """
    Ensemble model: text embeddings + product type → MLP → length prediction.
    """
    
    def __init__(self, config: Config, num_product_types: int):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        
        # Product type embedding
        self.product_emb = nn.Embedding(
            num_product_types, 
            config.model.product_type_emb_dim,
            padding_idx=0,
        )
        
        # MLP head
        input_dim = config.embeddings.total_dim + config.model.product_type_emb_dim
        self.head = MLPHead(
            input_dim=input_dim,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
            use_batch_norm=config.model.use_batch_norm,
        )
        
        nn.init.normal_(self.product_emb.weight, mean=0, std=0.02)
        
    def forward(self, text_embedding: torch.Tensor, product_type: torch.Tensor) -> torch.Tensor:
        type_emb = self.product_emb(product_type)
        combined = torch.cat([text_embedding, type_emb], dim=-1)
        return self.head(combined)
    
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss in log space."""
        pred_log = torch.log1p(F.relu(pred))
        target_log = torch.log1p(target)
        
        if self.config.training.loss_fn == "mse":
            return F.mse_loss(pred_log, target_log)
        elif self.config.training.loss_fn == "huber":
            return F.huber_loss(pred_log, target_log, delta=self.config.training.huber_delta)
        elif self.config.training.loss_fn == "mape":
            return torch.mean(torch.abs(pred - target) / (target + 1.0))
        else:
            raise ValueError(f"Unknown loss: {self.config.training.loss_fn}")
    
    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute evaluation metrics."""
        with torch.no_grad():
            pred_clipped = F.relu(pred) + 1e-6
            mape = torch.mean(torch.abs((target - pred_clipped) / target)) * 100
            rmsle = torch.sqrt(torch.mean((torch.log1p(pred_clipped) - torch.log1p(target)) ** 2))
        return {"mape": mape, "rmsle": rmsle}
    
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        pred = self(batch["text_embedding"], batch["product_type"])
        target = batch["target"]
        
        loss = self._compute_loss(pred, target)
        metrics = self._compute_metrics(pred, target)
        
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
        return self(batch["text_embedding"], batch["product_type"])
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.config.training.warmup_ratio * total_steps)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

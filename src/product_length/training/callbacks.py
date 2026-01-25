"""Custom Lightning callbacks for visualization and metric tracking."""

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import wandb


class SamplePredictionCallback(Callback):
    """Logs sample predictions to W&B for visual inspection."""
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.val_dataloaders:
            return
            
        preds, targets, types = [], [], []
        pl_module.eval()
        device = pl_module.device
        
        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                knn = batch.get("knn_features")
                pred = pl_module(
                    batch["text_embedding"].to(device),
                    batch["product_type"].to(device),
                    knn_features=knn.to(device) if knn is not None else None,
                )
                preds.append(pred.cpu().numpy())
                targets.append(batch["target"].numpy())
                types.append(batch["product_type"].numpy())
                
                if sum(len(p) for p in preds) >= self.num_samples * 2:
                    break
        
        preds, targets, types = np.concatenate(preds), np.concatenate(targets), np.concatenate(types)
        idx = np.random.choice(len(preds), min(self.num_samples, len(preds)), replace=False)
        
        data = [
            [targets[i], max(preds[i], 1e-6), abs(targets[i] - preds[i]), abs(targets[i] - preds[i]) / targets[i] * 100, int(types[i])]
            for i in idx
        ]
        table = wandb.Table(columns=["true", "pred", "abs_error", "pct_error", "product_type"], data=data)
        wandb.log({
            "val_samples": table,
            "epoch": trainer.current_epoch,
            "pred_vs_true": wandb.plot.scatter(table, "true", "pred", title=f"Epoch {trainer.current_epoch}")
        })


class MetricHistoryCallback(Callback):
    """Tracks MAPE history across epochs."""
    
    def __init__(self):
        self.train_mape: list[float] = []
        self.val_mape: list[float] = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        if "train_mape" in trainer.callback_metrics:
            self.train_mape.append(trainer.callback_metrics["train_mape"].item())
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_mape" in trainer.callback_metrics:
            self.val_mape.append(trainer.callback_metrics["val_mape"].item())

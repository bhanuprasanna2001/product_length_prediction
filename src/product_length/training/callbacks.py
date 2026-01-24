"""
Training Callbacks
==================
Custom callbacks for logging, visualization, and monitoring.
"""

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
            
        all_preds, all_targets, all_types = [], [], []
        pl_module.eval()
        device = pl_module.device
        
        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                pred = pl_module(
                    batch["text_embedding"].to(device),
                    batch["product_type"].to(device),
                )
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch["target"].numpy())
                all_types.append(batch["product_type"].numpy())
                
                if sum(len(p) for p in all_preds) >= self.num_samples * 2:
                    break
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        types = np.concatenate(all_types)
        
        # Random sample
        n = min(self.num_samples, len(preds))
        idx = np.random.choice(len(preds), n, replace=False)
        
        # Create W&B table
        columns = ["true", "pred", "abs_error", "pct_error", "product_type"]
        data = []
        for i in idx:
            true_val = targets[i]
            pred_val = max(preds[i], 1e-6)
            abs_err = abs(true_val - pred_val)
            pct_err = abs_err / true_val * 100
            data.append([true_val, pred_val, abs_err, pct_err, int(types[i])])
            
        table = wandb.Table(columns=columns, data=data)
        
        wandb.log({
            "val_samples": table,
            "epoch": trainer.current_epoch,
            "pred_vs_true": wandb.plot.scatter(
                table, "true", "pred",
                title=f"Predictions vs True (Epoch {trainer.current_epoch})"
            )
        })


class MetricHistoryCallback(Callback):
    """Tracks metric history for analysis."""
    
    def __init__(self):
        self.train_mape = []
        self.val_mape = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        if "train_mape" in trainer.callback_metrics:
            self.train_mape.append(trainer.callback_metrics["train_mape"].item())
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_mape" in trainer.callback_metrics:
            self.val_mape.append(trainer.callback_metrics["val_mape"].item())

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
import numpy as np


class ProductLengthModel(pl.LightningModule):
    def __init__(self, config, num_product_types):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Text encoder (frozen initially)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Product type embedding
        self.product_emb = nn.Embedding(num_product_types, config.product_type_emb_dim)
        
        # MLP head
        input_dim = self.text_dim + config.product_type_emb_dim
        layers = []
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_mask, product_type):
        # Text encoding (mean pooling)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = (text_out.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
        text_emb = text_emb / attention_mask.sum(-1, keepdim=True)
        
        # Product type embedding
        type_emb = self.product_emb(product_type)
        
        # Concatenate and predict
        combined = torch.cat([text_emb, type_emb], dim=-1)
        return self.head(combined).squeeze(-1)
    
    def _step(self, batch, stage):
        pred = self(batch['input_ids'], batch['attention_mask'], batch['product_type'])
        target = batch['target']
        
        # Log-space MSE (approximates RMSLE)
        pred_log = torch.log1p(torch.clamp(pred, min=0))
        target_log = torch.log1p(target)
        loss = nn.functional.mse_loss(pred_log, target_log)
        
        # Metrics
        with torch.no_grad():
            mape = torch.mean(torch.abs((target - pred) / target)) * 100
            rmsle = torch.sqrt(loss)
        
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_mape', mape, prog_bar=True)
        self.log(f'{stage}_rmsle', rmsle, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    
    def configure_optimizers(self):
        # Different LR for encoder vs head
        encoder_params = list(self.text_encoder.parameters())
        other_params = list(self.product_emb.parameters()) + list(self.head.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.config.lr * 0.1},
            {'params': other_params, 'lr': self.config.lr}
        ], weight_decay=self.config.weight_decay)
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.config.lr * 0.1, self.config.lr],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

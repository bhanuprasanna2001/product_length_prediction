import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

from src.config import Config
from src.data.dataset import ProductDataModule
from src.model import ProductLengthModel


def main():
    config = Config()
    pl.seed_everything(config.seed)
    
    # Data
    dm = ProductDataModule(config)
    dm.setup()
    
    # Model
    model = ProductLengthModel(config, dm.num_product_types)
    
    # W&B Logger
    wandb_logger = WandbLogger(
        project="amazon-product-length",
        name="text_encoder_v1",
        config=config.__dict__
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch}-{val_rmsle:.4f}",
            monitor="val_rmsle",
            mode="min",
            save_top_k=1
        ),
        EarlyStopping(monitor="val_rmsle", patience=3, mode="min"),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        val_check_interval=0.5,
        log_every_n_steps=50
    )
    
    trainer.fit(model, dm)
    wandb.finish()


if __name__ == "__main__":
    main()

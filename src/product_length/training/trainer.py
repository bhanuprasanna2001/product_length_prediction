"""
Training Pipeline
=================
Main training loop with callbacks, logging, and evaluation.
"""

import pickle
import numpy as np
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import wandb

from ..config import Config
from ..data import EmbeddingDataModule
from ..models import EnsembleModel
from ..inference import create_postprocessor, collect_predictions
from ..utils import mape, rmsle, evaluate_predictions
from .callbacks import SamplePredictionCallback, MetricHistoryCallback


def train(config: Config) -> tuple[str, float]:
    """
    Main training function.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (best_checkpoint_path, best_val_mape)
    """
    pl.seed_everything(config.system.seed)
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Active embeddings: {config.embeddings.active}")
    print(f"Total embedding dim: {config.embeddings.total_dim}")
    print(f"Loss function: {config.training.loss_fn}")
    print("=" * 60)
    
    # Data
    print("\nLoading data...")
    dm = EmbeddingDataModule(config)
    dm.setup()
    
    # Model
    model = EnsembleModel(config, dm.num_product_types)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Logger
    run_name = config.logging.run_name or f"ensemble_{config.training.loss_fn}_{datetime.now()}"
    wandb_logger = WandbLogger(
        project=config.logging.project,
        name=run_name,
        config=config.to_dict(),
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoints.dir,
            filename="ensemble-{epoch:02d}-{val_mape:.2f}",
            monitor=config.checkpoints.monitor,
            mode=config.checkpoints.mode,
            save_top_k=config.checkpoints.save_top_k,
        ),
        EarlyStopping(
            monitor=config.checkpoints.monitor,
            patience=config.training.patience,
            mode=config.checkpoints.mode,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SamplePredictionCallback(num_samples=100),
        MetricHistoryCallback(),
    ]
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        val_check_interval=config.logging.val_check_interval,
        log_every_n_steps=config.logging.log_every_n_steps,
        precision=config.training.precision,
    )
    
    print("\nStarting training...")
    trainer.fit(model, dm)
    
    # -------------------------------------------------------------------------
    # Post-Training Evaluation
    # -------------------------------------------------------------------------
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_checkpoint}")
    
    # Load best model
    model = EnsembleModel.load_from_checkpoint(best_checkpoint, config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Collect validation predictions
    print("\nEvaluating on validation set...")
    val_preds, val_targets, val_types = collect_predictions(model, dm.val_dataloader(), device)
    
    # Fit post-processor
    print("\nFitting post-processor...")
    postprocessor = create_postprocessor(
        train_targets=np.array(dm.train_ds.targets),
        train_product_types=np.array(dm.train_ds.product_types),
        val_preds=val_preds,
        val_targets=val_targets,
        calibration_method=config.postprocessing.calibration_method,
    )
    
    # Evaluate with post-processing
    val_results = evaluate_predictions(
        val_preds, val_targets, val_types, postprocessor, "validation"
    )
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    test_preds, test_targets, test_types = collect_predictions(model, dm.test_dataloader(), device)
    test_results = evaluate_predictions(
        test_preds, test_targets, test_types, postprocessor, "test"
    )
    
    # Log final results
    wandb.log({
        "final_val_mape_raw": val_results["raw_mape"],
        "final_val_mape_pp": val_results["pp_mape"],
        "final_test_mape_raw": test_results["raw_mape"],
        "final_test_mape_pp": test_results["pp_mape"],
    })
    
    # Save post-processor
    pp_path = config.checkpoints.dir / "postprocessor.pkl"
    with open(pp_path, "wb") as f:
        pickle.dump(postprocessor, f)
    print(f"Post-processor saved: {pp_path}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Test MAPE (raw):  {test_results['raw_mape']:.2f}%")
    print(f"Test MAPE (post): {test_results['pp_mape']:.2f}%")
    print("=" * 60)
    
    wandb.finish()
    
    return best_checkpoint, test_results["pp_mape"]

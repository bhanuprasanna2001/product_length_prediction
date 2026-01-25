"""Training pipeline with Lightning callbacks, W&B logging, and evaluation."""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ..config import Config
from ..constants import DEFAULT_SAMPLE_SIZE
from ..data import EmbeddingDataModule
from ..inference import collect_predictions, create_postprocessor, PostProcessor
from ..models import EnsembleModel
from ..utils import evaluate_predictions, get_device
from ..utils.logging import get_logger, print_header, print_section, print_success, print_warning, console
from .callbacks import MetricHistoryCallback, SamplePredictionCallback

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Training run results."""
    best_checkpoint: str
    test_mape_raw: float
    test_mape_postprocessed: float
    val_mape_raw: float
    val_mape_postprocessed: float
    
    @property
    def best_mape(self) -> float:
        return min(self.test_mape_raw, self.test_mape_postprocessed)


class CallbackFactory:
    """Creates Lightning callbacks for training."""
    
    @staticmethod
    def create_all(config: Config) -> list[Callback]:
        return [
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
            SamplePredictionCallback(num_samples=DEFAULT_SAMPLE_SIZE),
            MetricHistoryCallback(),
        ]


class TrainerFactory:
    """Creates Lightning Trainer."""
    
    @staticmethod
    def create(config: Config, wandb_logger: WandbLogger, callbacks: list[Callback]) -> pl.Trainer:
        return pl.Trainer(
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


def create_wandb_logger(config: Config, run_name: str | None = None) -> WandbLogger:
    """Create W&B logger for experiment tracking."""
    name = run_name or f"ensemble_{config.training.loss_fn}_{datetime.now()}"
    return WandbLogger(project=config.logging.project, name=name, config=config.to_dict())


class ModelEvaluator:
    """Evaluates trained models on validation and test sets."""
    
    def __init__(self, config: Config, data_module: EmbeddingDataModule):
        self.config = config
        self.dm = data_module
        self.device = get_device()
    
    def load_best_model(self, checkpoint_path: str) -> EnsembleModel:
        model = EnsembleModel.load_from_checkpoint(checkpoint_path, config=self.config, knn_dim=self.dm.knn_dim)
        model.eval()
        return model.to(self.device)
    
    def evaluate(self, model: EnsembleModel) -> dict[str, Any]:
        """Evaluate on validation and test sets, returning results and fitted postprocessor."""
        # Validation predictions
        print("\nEvaluating on validation set...")
        val_result = collect_predictions(model, self.dm.val_dataloader(), self.device)
        
        # Fit postprocessor
        print("\nFitting post-processor...")
        assert self.dm.train_ds is not None
        assert val_result.targets is not None, "Validation targets required for postprocessor fitting"
        
        postprocessor = create_postprocessor(
            train_targets=np.array(self.dm.train_ds.targets),
            train_product_types=np.array(self.dm.train_ds.product_types),
            val_preds=val_result.predictions,
            val_targets=val_result.targets,
        )
        
        # Evaluate both sets
        val_results = evaluate_predictions(
            val_result.predictions, val_result.targets, val_result.product_types, postprocessor, "validation"
        )
        
        print("\nEvaluating on test set...")
        test_result = collect_predictions(model, self.dm.test_dataloader(), self.device)
        
        assert test_result.targets is not None, "Test targets required for evaluation"
        test_results = evaluate_predictions(
            test_result.predictions, test_result.targets, test_result.product_types, postprocessor, "test"
        )
        
        # Save postprocessor
        pp_path = self.config.checkpoints.dir / "postprocessor.pkl"
        with open(pp_path, "wb") as f:
            pickle.dump(postprocessor, f)
        print(f"Post-processor saved: {pp_path}")
        
        return {"validation": val_results, "test": test_results, "postprocessor": postprocessor}


# =============================================================================
# Training Orchestrator
# =============================================================================

def _print_config_summary(config: Config) -> None:
    print_header("TRAINING PIPELINE")
    console.print(f"  [cyan]Embeddings:[/] {', '.join(config.embeddings.active)}")
    console.print(f"  [cyan]Total dim:[/] {config.embeddings.total_dim}")
    console.print(f"  [cyan]Loss:[/] {config.training.loss_fn}")
    console.print(f"  [cyan]Batch size:[/] {config.training.batch_size}")
    console.print(f"  [cyan]Learning rate:[/] {config.training.lr}")
    console.print(f"  [cyan]Epochs:[/] {config.training.epochs}")
    knn_status = f"k={config.features.knn_k}, embedding={config.features.knn_embedding}" if config.features.use_knn else "disabled"
    console.print(f"  [cyan]KNN:[/] {knn_status}")


def _print_training_summary(result: TrainingResult) -> None:
    print_header("TRAINING COMPLETE")
    console.print(f"  [cyan]Best checkpoint:[/] [path]{result.best_checkpoint}[/]")
    console.print(f"\n  [metric]Test MAPE (raw):[/]  {result.test_mape_raw:.2f}%")
    console.print(f"  [metric]Test MAPE (post):[/] {result.test_mape_postprocessed:.2f}%")
    
    diff = result.test_mape_raw - result.test_mape_postprocessed
    if diff > 0:
        print_success(f"Post-processing improved by {diff:.2f}%")
    elif diff < 0:
        print_warning(f"Post-processing degraded by {-diff:.2f}%")


def train(config: Config) -> tuple[str, float]:
    """
    Run training pipeline: setup → train → evaluate → cleanup.
    Returns (best_checkpoint_path, best_test_mape).
    """
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    pl.seed_everything(config.system.seed)
    _print_config_summary(config)
    
    # Setup
    print_section("Loading Data")
    data_module = EmbeddingDataModule(config)
    data_module.setup()
    
    model = EnsembleModel(config, data_module.num_product_types, knn_dim=data_module.knn_dim, use_log_target=config.features.use_log_target)
    console.print(f"  [cyan]Model parameters:[/] {sum(p.numel() for p in model.parameters()):,}")
    if config.features.use_log_target:
        console.print("  [yellow]Log-target transform:[/] enabled")
    
    wandb_logger = create_wandb_logger(config, config.logging.run_name)
    callbacks = CallbackFactory.create_all(config)
    trainer = TrainerFactory.create(config, wandb_logger, callbacks)
    
    # Train
    print_section("Training")
    trainer.fit(model, data_module)
    
    checkpoint_callback: ModelCheckpoint | None = None
    for cb in trainer.callbacks or []:
        if isinstance(cb, ModelCheckpoint):
            checkpoint_callback = cb
            break
    
    assert checkpoint_callback is not None, "ModelCheckpoint callback not found"
    best_checkpoint = str(checkpoint_callback.best_model_path)
    print_success(f"Best checkpoint: {best_checkpoint}")
    
    # Evaluate
    print_section("Evaluation")
    evaluator = ModelEvaluator(config, data_module)
    best_model = evaluator.load_best_model(best_checkpoint)
    eval_results = evaluator.evaluate(best_model)
    
    # Log and summarize
    wandb.log({
        "final_val_mape_raw": eval_results["validation"]["raw_mape"],
        "final_val_mape_pp": eval_results["validation"]["pp_mape"],
        "final_test_mape_raw": eval_results["test"]["raw_mape"],
        "final_test_mape_pp": eval_results["test"]["pp_mape"],
    })
    
    result = TrainingResult(
        best_checkpoint=best_checkpoint,
        test_mape_raw=eval_results["test"]["raw_mape"],
        test_mape_postprocessed=eval_results["test"]["pp_mape"],
        val_mape_raw=eval_results["validation"]["raw_mape"],
        val_mape_postprocessed=eval_results["validation"]["pp_mape"],
    )
    
    _print_training_summary(result)
    wandb.finish()
    
    return result.best_checkpoint, result.best_mape

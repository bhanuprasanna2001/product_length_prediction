#!/usr/bin/env python
"""Train ensemble model with Hydra configuration management."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.product_length.training import train
from src.product_length.utils.logging import setup_logging, print_header, print_success


def config_to_dataclass(cfg: DictConfig):
    """Convert Hydra DictConfig to Config dataclass."""
    from src.product_length.config import (
        Config, DataConfig, EmbeddingsConfig, EmbeddingModelConfig,
        ModelConfig, TrainingConfig, PostprocessingConfig,
        LoggingConfig, CheckpointConfig, FeaturesConfig, SystemConfig
    )
    
    embedding_models = {
        name: EmbeddingModelConfig(name=m.name, dim=m.dim)
        for name, m in cfg.embeddings.models.items()
    }
    
    return Config(
        data=DataConfig(
            train_path=Path(cfg.data.train_path),
            test_path=Path(cfg.data.test_path),
            embedding_dir=Path(cfg.data.embedding_dir),
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
        ),
        embeddings=EmbeddingsConfig(
            models=embedding_models,
            active=list(cfg.embeddings.active),
            batch_size=cfg.embeddings.batch_size,
        ),
        model=ModelConfig(
            product_type_emb_dim=cfg.model.product_type_emb_dim,
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            use_batch_norm=cfg.model.use_batch_norm,
        ),
        training=TrainingConfig(
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            epochs=cfg.training.epochs,
            warmup_ratio=cfg.training.warmup_ratio,
            loss_fn=cfg.training.loss_fn,
            huber_delta=cfg.training.huber_delta,
            patience=cfg.training.patience,
            gradient_clip_val=cfg.training.gradient_clip_val,
            precision=cfg.training.precision,
        ),
        postprocessing=PostprocessingConfig(
            use_snapping=cfg.postprocessing.use_snapping,
            snap_by_type=cfg.postprocessing.snap_by_type,
        ),
        logging=LoggingConfig(
            project=cfg.logging.project,
            run_name=cfg.logging.run_name,
            log_every_n_steps=cfg.logging.log_every_n_steps,
            val_check_interval=cfg.logging.val_check_interval,
        ),
        checkpoints=CheckpointConfig(
            dir=Path(cfg.checkpoints.dir),
            save_top_k=cfg.checkpoints.save_top_k,
            monitor=cfg.checkpoints.monitor,
            mode=cfg.checkpoints.mode,
        ),
        features=FeaturesConfig(
            use_knn=cfg.features.use_knn,
            knn_k=cfg.features.knn_k,
            knn_embeddings=list(cfg.features.knn_embeddings),
            knn_ensemble=cfg.features.knn_ensemble,
            use_log_target=cfg.features.use_log_target,
        ),
        system=SystemConfig(
            seed=cfg.system.seed,
            num_workers=cfg.system.num_workers,
        ),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train model and return best MAPE for Hydra sweeps."""
    setup_logging()
    
    print_header("Product Length Prediction - Training")
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    config = config_to_dataclass(cfg)
    best_checkpoint, best_mape = train(config)
    
    print_success(f"\n✅ Training complete!")
    print(f"   Best checkpoint: {best_checkpoint}")
    print(f"   Best val MAPE: {best_mape:.2f}%")
    
    return best_mape


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train the product length prediction model.

Usage:
    python scripts/train.py                              # with default.yaml
    python scripts/train.py --config configs/custom.yaml # custom config
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.product_length.config import Config
from src.product_length.dataset import EmbeddingDataset
from src.product_length.losses import mape_numpy, rmsle_numpy, score_numpy, EPSILON
from src.product_length.model import EnsembleModel
from src.product_length.postprocessing import create_snapper


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_data(cfg: Config) -> dict:
    """Load CSV metadata, memory-mapped embeddings, and KNN features."""
    print("Loading data...")
    train_df = pd.read_csv(cfg.data_dir / "total_sentence_train.csv")
    test_df = pd.read_csv(cfg.data_dir / "total_sentence_test.csv")
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Load embeddings (memory-mapped)
    embeddings = {}
    for name in cfg.active_embeddings:
        path = cfg.embedding_dir / f"{name}_train.npy"
        embeddings[name] = np.load(path, mmap_mode="r")
        print(f"  Embedding {name}: {embeddings[name].shape}")

    # Load KNN features (optional)
    knn_features = None
    if cfg.use_knn:
        knn_features = {}
        for name in cfg.knn_embeddings:
            path = cfg.knn_dir / f"knn_k{cfg.knn_k}_{name}_train.npy"
            knn_features[name] = np.load(path, mmap_mode="r")
            print(f"  KNN {name}: {knn_features[name].shape}")

    # Build product type mapping (0 = unknown)
    all_types = pd.concat([train_df["PRODUCT_TYPE_ID"], test_df["PRODUCT_TYPE_ID"]]).unique()
    type_to_idx = {t: i + 1 for i, t in enumerate(sorted(all_types))}

    return {
        "embeddings": embeddings,
        "knn_features": knn_features,
        "targets": train_df["PRODUCT_LENGTH"].values.astype(np.float32),
        "product_types": train_df["PRODUCT_TYPE_ID"].map(type_to_idx).values,
        "num_product_types": len(type_to_idx) + 1,
    }


def create_dataloaders(cfg: Config, data: dict) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Split data and create train/val/test DataLoaders."""
    n = len(data["targets"])
    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(n)

    train_end = int(cfg.train_ratio * n)
    val_end = train_end + int(cfg.val_ratio * n)
    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    print(f"  Splits — train: {len(splits['train']):,}  val: {len(splits['val']):,}  test: {len(splits['test']):,}")

    def make_ds(split_idx):
        return EmbeddingDataset(
            embeddings=data["embeddings"],
            indices=split_idx,
            product_types=data["product_types"][split_idx],
            targets=data["targets"][split_idx],
            knn_features=data["knn_features"],
            use_log_target=cfg.use_log_target,
        )

    train_ds, val_ds, test_ds = make_ds(splits["train"]), make_ds(splits["val"]), make_ds(splits["test"])
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, **loader_kwargs),
        splits,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_predictions(model, dataloader, device, use_log_target: bool = False):
    """Run inference and collect predictions, targets, and product types."""
    model.eval()
    preds, targets, ptypes = [], [], []

    for batch in dataloader:
        emb = batch["text_embedding"].to(device)
        pt = batch["product_type"].to(device)
        knn = batch.get("knn_features")
        if knn is not None:
            knn = knn.to(device)

        pred = model(emb, pt, knn)
        pred = torch.relu(pred) + EPSILON
        if use_log_target:
            pred = torch.expm1(pred)

        preds.append(pred.cpu().numpy())
        if "target" in batch:
            t = batch["target"]
            if use_log_target:
                t = torch.expm1(t)
            targets.append(t.numpy())
        ptypes.append(batch["product_type"].numpy())

    return {
        "preds": np.concatenate(preds),
        "targets": np.concatenate(targets) if targets else None,
        "product_types": np.concatenate(ptypes),
    }


def evaluate(results: dict, snapper, stage: str = "test") -> dict[str, float]:
    """Evaluate raw and snapped predictions, printing a summary."""
    raw_mape = mape_numpy(results["targets"], results["preds"])
    raw_rmsle = rmsle_numpy(results["targets"], results["preds"])
    snapped = snapper.process(results["preds"], results["product_types"])
    snap_mape = mape_numpy(results["targets"], snapped)
    snap_rmsle = rmsle_numpy(results["targets"], snapped)

    print(f"\n{'=' * 55}")
    print(f"  {stage.upper()} RESULTS  (loss={cfg.loss_fn}, log_target={cfg.use_log_target})")
    print(f"{'=' * 55}")
    print(f"  Raw   MAPE: {raw_mape:.2f}%   RMSLE: {raw_rmsle:.4f}")
    print(f"  Snap  MAPE: {snap_mape:.2f}%   RMSLE: {snap_rmsle:.4f}")
    print(f"  Improvement: {raw_mape - snap_mape:+.2f}%")
    print(f"  Score: {score_numpy(results['targets'], snapped):.2f}")

    return {"raw_mape": raw_mape, "snap_mape": snap_mape, "raw_rmsle": raw_rmsle, "snap_rmsle": snap_rmsle}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train product length prediction model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    global cfg
    cfg = Config.from_yaml(args.config)
    pl.seed_everything(cfg.seed)

    print(f"\n{'=' * 55}")
    print(f"  TRAINING PIPELINE")
    print(f"{'=' * 55}")
    print(f"  Embeddings:  {cfg.active_embeddings}")
    print(f"  Total dim:   {cfg.total_embedding_dim}")
    print(f"  KNN:         {'k=' + str(cfg.knn_k) + ' (' + ', '.join(cfg.knn_embeddings) + ')' if cfg.use_knn else 'disabled'}")
    print(f"  Loss:        {cfg.loss_fn}  |  Log-target: {cfg.use_log_target}")
    print(f"  Architecture: {cfg.hidden_dims}")
    print(f"  LR: {cfg.lr}  |  Batch: {cfg.batch_size}  |  Epochs: {cfg.epochs}")
    print(f"{'=' * 55}\n")

    # Load data
    data = load_data(cfg)
    train_loader, val_loader, test_loader, splits = create_dataloaders(cfg, data)

    # Create model
    model = EnsembleModel(cfg, data["num_product_types"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}\n")

    # W&B logger
    run_name = cfg.wandb_run_name or f"ensemble_{cfg.loss_fn}_{datetime.now().strftime('%m%d_%H%M')}"
    wandb_logger = WandbLogger(project=cfg.wandb_project, name=run_name, config=cfg.to_dict())

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="best-{epoch:02d}-{val_mape:.2f}",
            monitor="val_mape",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_mape", patience=cfg.patience, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint and evaluate
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    model = EnsembleModel.load_from_checkpoint(best_ckpt, config=cfg, num_product_types=data["num_product_types"])
    model.eval()
    device = next(model.parameters()).device

    # Build snapper from training data
    snapper = create_snapper(data["targets"][splits["train"]], data["product_types"][splits["train"]])

    # Evaluate validation and test
    val_results = collect_predictions(model, val_loader, device, cfg.use_log_target)
    val_metrics = evaluate(val_results, snapper, "validation")

    test_results = collect_predictions(model, test_loader, device, cfg.use_log_target)
    test_metrics = evaluate(test_results, snapper, "test")

    # Log final metrics to W&B
    wandb.log({
        "final_val_mape": val_metrics["snap_mape"],
        "final_test_mape": test_metrics["snap_mape"],
        "final_test_score": score_numpy(test_results["targets"], snapper.process(test_results["preds"], test_results["product_types"])),
    })
    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()

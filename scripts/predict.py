#!/usr/bin/env python
"""Generate submission predictions from a trained checkpoint.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best-epoch=12-val_mape=51.78.ckpt
    python scripts/predict.py --checkpoint best.ckpt --output submission.csv --no-snap
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.product_length.config import Config
from src.product_length.dataset import EmbeddingDataset
from src.product_length.losses import EPSILON
from src.product_length.model import EnsembleModel
from src.product_length.postprocessing import create_snapper


def main():
    parser = argparse.ArgumentParser(description="Generate submission predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--no-snap", action="store_true", help="Disable snapping post-processing")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # Load training data for product type mapping + snapper
    train_df = pd.read_csv(cfg.data_dir / "total_sentence_train.csv")
    test_df = pd.read_csv(cfg.data_dir / "total_sentence_test.csv")

    all_types = pd.concat([train_df["PRODUCT_TYPE_ID"], test_df["PRODUCT_TYPE_ID"]]).unique()
    type_to_idx = {t: i + 1 for i, t in enumerate(sorted(all_types))}

    # Load model
    model = EnsembleModel.load_from_checkpoint(
        args.checkpoint, config=cfg, num_product_types=len(type_to_idx) + 1
    )
    model.eval().to(device)

    # Load test embeddings
    embeddings = {
        name: np.load(cfg.embedding_dir / f"{name}_test.npy", mmap_mode="r")
        for name in cfg.active_embeddings
    }

    knn_features = None
    if cfg.use_knn:
        knn_features = {
            name: np.load(cfg.knn_dir / f"knn_k{cfg.knn_k}_{name}_test.npy", mmap_mode="r")
            for name in cfg.knn_embeddings
        }

    test_types = test_df["PRODUCT_TYPE_ID"].map(lambda x: type_to_idx.get(x, 0)).values
    test_ds = EmbeddingDataset(
        embeddings=embeddings,
        indices=np.arange(len(test_df)),
        product_types=test_types,
        knn_features=knn_features,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)

    # Generate predictions
    preds_list, types_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            emb = batch["text_embedding"].to(device)
            pt = batch["product_type"].to(device)
            knn = batch.get("knn_features")
            if knn is not None:
                knn = knn.to(device)

            pred = model(emb, pt, knn)
            pred = torch.relu(pred) + EPSILON
            if cfg.use_log_target:
                pred = torch.expm1(pred)

            preds_list.append(pred.cpu().numpy())
            types_list.append(batch["product_type"].numpy())

    preds = np.concatenate(preds_list)
    product_types = np.concatenate(types_list)

    print(f"Raw predictions: {len(preds):,} samples, range [{preds.min():.2f}, {preds.max():.2f}]")

    # Post-processing
    if not args.no_snap:
        train_targets = train_df["PRODUCT_LENGTH"].values.astype(np.float32)
        train_types = train_df["PRODUCT_TYPE_ID"].map(type_to_idx).values
        snapper = create_snapper(train_targets, train_types)
        preds = snapper.process(preds, product_types)
        print(f"Snapped range: [{preds.min():.2f}, {preds.max():.2f}]")

    preds = np.maximum(preds, EPSILON)

    # Save
    submission = pd.DataFrame({"PRODUCT_ID": test_df["PRODUCT_ID"].values, "PRODUCT_LENGTH": preds})
    submission.to_csv(args.output, index=False)
    print(f"Saved: {args.output} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()

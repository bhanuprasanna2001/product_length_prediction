#!/usr/bin/env python
"""Extract KNN retrieval features from pre-computed embeddings.

For each sample, finds K nearest neighbors in embedding space and computes
statistical features (mean, std, median, min, max) from their product lengths.

Usage:
    python scripts/extract_knn_features.py --embedding minilm
    python scripts/extract_knn_features.py --embeddings minilm mpnet distiluse e5small --k 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.product_length.config import Config
from src.product_length.embeddings import build_faiss_index, extract_knn_features


def process_embedding(
    name: str, cfg: Config, lengths: np.ndarray, k: int
):
    """Build FAISS index and extract KNN features for train (and test if available)."""
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")

    train_emb = np.load(cfg.embedding_dir / f"{name}_train.npy")
    print(f"  Train embeddings: {train_emb.shape}")

    # Build index from training embeddings
    index = build_faiss_index(train_emb)

    # Train features (exclude self-match)
    print(f"  Extracting train KNN features (k={k})...")
    train_features = extract_knn_features(index, train_emb, lengths, k, is_train=True)
    out_path = cfg.knn_dir / f"knn_k{k}_{name}_train.npy"
    np.save(out_path, train_features)
    print(f"  Saved: {out_path} {train_features.shape}")

    # Test features (if test embeddings exist)
    test_path = cfg.embedding_dir / f"{name}_test.npy"
    if test_path.exists():
        test_emb = np.load(test_path)
        print(f"  Test embeddings: {test_emb.shape}")
        print(f"  Extracting test KNN features...")
        test_features = extract_knn_features(index, test_emb, lengths, k, is_train=False)
        out_path = cfg.knn_dir / f"knn_k{k}_{name}_test.npy"
        np.save(out_path, test_features)
        print(f"  Saved: {out_path} {test_features.shape}")

    # Print feature statistics
    for i, stat in enumerate(["mean", "std", "median", "min", "max"]):
        print(f"    knn_{stat:7s}: mean={train_features[:, i].mean():8.2f}  std={train_features[:, i].std():8.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract KNN retrieval features from pre-computed embeddings",
        epilog="Example: python scripts/extract_knn_features.py --models minilm mpnet distiluse e5small",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Embedding model names to process (e.g. minilm mpnet). Defaults to knn_embeddings in config.",
    )
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.knn_dir.mkdir(parents=True, exist_ok=True)
    models = args.models or cfg.knn_embeddings

    print(f"{'=' * 55}")
    print(f"  KNN FEATURE EXTRACTION")
    print(f"  Models: {models}")
    print(f"  K: {args.k}")
    print(f"{'=' * 55}")

    # Load training lengths
    lengths = pd.read_csv(
        cfg.data_dir / "total_sentence_train.csv"
    )["PRODUCT_LENGTH"].values.astype(np.float32)
    print(f"  Training lengths: {len(lengths):,} samples, range [{lengths.min():.1f}, {lengths.max():.1f}]")

    for name in models:
        process_embedding(name, cfg, lengths, args.k)

    print(f"\nDone.")


if __name__ == "__main__":
    main()

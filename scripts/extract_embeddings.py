#!/usr/bin/env python
"""Pre-compute sentence embeddings from multilingual transformer models.

Usage:
    python scripts/extract_embeddings.py --split train
    python scripts/extract_embeddings.py --split both --models minilm mpnet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.product_length.config import Config, EMBEDDING_REGISTRY
from src.product_length.embeddings import extract_embeddings, save_embeddings


def process_split(cfg: Config, split: str, models: list[str]):
    """Extract and save embeddings for one data split."""
    csv_path = cfg.data_dir / f"total_sentence_{split}.csv"
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    texts = df["TOTAL_SENTENCE"].fillna("").tolist()
    print(f"  {len(texts):,} samples")

    for model_key in models:
        out_path = cfg.embedding_dir / f"{model_key}_{split}.npy"
        if out_path.exists():
            print(f"  [SKIP] {model_key} — already exists at {out_path}")
            continue

        print(f"\n  Extracting: {model_key}")
        embeddings = extract_embeddings(texts, model_key, batch_size=128)
        save_embeddings(embeddings, model_key, split, cfg.embedding_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract sentence embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", required=True, choices=["train", "test", "both"])
    parser.add_argument("--models", nargs="+", default=None, help=f"Models: {list(EMBEDDING_REGISTRY)}")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    models = args.models or cfg.active_embeddings

    print(f"{'=' * 55}")
    print(f"  EMBEDDING EXTRACTION")
    print(f"  Models: {models}")
    print(f"  Split:  {args.split}")
    print(f"{'=' * 55}")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        process_split(cfg, split, models)

    print(f"\nDone.")


if __name__ == "__main__":
    main()

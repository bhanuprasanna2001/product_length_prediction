#!/usr/bin/env python
"""Pre-compute embeddings from multilingual models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.product_length.config import load_config
from src.product_length.constants import EMBEDDING_MODELS
from src.product_length.utils.embeddings import extract_embeddings, save_embeddings, save_metadata


def process_split(config, split: str, models: list[str]):
    """Process a single data split."""
    data_path = config.data.train_path if split == "train" else config.data.test_path
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples")
    
    texts = df["TOTAL_SENTENCE"].fillna("").tolist()
    output_dir = config.data.embedding_dir
    
    for model_key in models:
        output_path = output_dir / f"{model_key}_{split}.npy"
        if output_path.exists():
            print(f"\n[SKIP] {model_key} already exists: {output_path}")
            continue
            
        print(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        embeddings = extract_embeddings(texts, model_key, batch_size=config.embeddings.batch_size)
        save_embeddings(embeddings, model_key, split, output_dir)
    
    metadata_path = output_dir / f"metadata_{split}.parquet"
    if not metadata_path.exists():
        save_metadata(df, split, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from multilingual models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test", "both"])
    parser.add_argument("--models", type=str, nargs="+", default=None, help=f"Models: {list(EMBEDDING_MODELS.keys())}")
    args = parser.parse_args()
    
    config = load_config(args.config)
    models = args.models or config.embeddings.active
    
    print(f"{'='*60}\nEMBEDDING EXTRACTION\n{'='*60}\nModels: {models}\nSplit: {args.split}\n{'='*60}")
    
    if args.split == "both":
        process_split(config, "train", models)
        process_split(config, "test", models)
    else:
        process_split(config, args.split, models)
    
    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == "__main__":
    main()

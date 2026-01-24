#!/usr/bin/env python
"""
Extract Embeddings
==================
Pre-compute embeddings from multilingual models.

Usage:
    python scripts/extract_embeddings.py --split train
    python scripts/extract_embeddings.py --split test
    python scripts/extract_embeddings.py --split both --models minilm mpnet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.product_length.config import load_config
from src.product_length.utils.embeddings import (
    EMBEDDING_MODELS,
    extract_embeddings,
    save_embeddings,
    save_metadata,
)


def process_split(config, split: str, models: list[str]):
    """Process a single data split."""
    # Load data
    data_path = config.data.train_path if split == "train" else config.data.test_path
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples")
    
    texts = df["TOTAL_SENTENCE"].fillna("").tolist()
    output_dir = config.data.embedding_dir
    
    # Process each model
    for model_key in models:
        output_path = output_dir / f"{model_key}_{split}.npy"
        
        if output_path.exists():
            print(f"\n[SKIP] {model_key} already exists: {output_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")
        
        embeddings = extract_embeddings(
            texts, 
            model_key, 
            batch_size=config.embeddings.batch_size,
        )
        save_embeddings(embeddings, model_key, split, output_dir)
    
    # Save metadata
    metadata_path = output_dir / f"metadata_{split}.parquet"
    if not metadata_path.exists():
        save_metadata(df, split, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from multilingual models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test", "both"])
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help=f"Models to run. Options: {list(EMBEDDING_MODELS.keys())}")
    args = parser.parse_args()
    
    config = load_config(args.config)
    models = args.models or config.embeddings.active
    
    print("=" * 60)
    print("EMBEDDING EXTRACTION")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Split: {args.split}")
    print("=" * 60)
    
    if args.split == "both":
        process_split(config, "train", models)
        process_split(config, "test", models)
    else:
        process_split(config, args.split, models)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract KNN features from multiple embeddings for ensemble approach."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss


def load_embedding(embedding_dir: Path, model: str, split: str) -> np.ndarray:
    path = embedding_dir / f"{model}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    return np.load(path)


def extract_knn_features_for_split(split: str, embeddings: np.ndarray, index: faiss.Index, lengths: np.ndarray, k: int, is_train: bool = False, batch_size: int = 50000) -> np.ndarray:
    """Extract KNN features for a split. For train, excludes self from neighbors."""
    n = len(embeddings)
    features = np.zeros((n, 5), dtype=np.float32)
    actual_k = k + 1 if is_train else k
    neighbor_slice = slice(1, None) if is_train else slice(None)
    
    for i in tqdm(range(0, n, batch_size), desc=f"{split} KNN"):
        end = min(i + batch_size, n)
        batch = embeddings[i:end].astype(np.float32).copy()
        faiss.normalize_L2(batch)
        
        _, indices = index.search(batch, actual_k)
        neighbor_lengths = lengths[indices[:, neighbor_slice]]
        
        features[i:end, 0] = neighbor_lengths.mean(axis=1)
        features[i:end, 1] = neighbor_lengths.std(axis=1)
        features[i:end, 2] = np.median(neighbor_lengths, axis=1)
        features[i:end, 3] = neighbor_lengths.min(axis=1)
        features[i:end, 4] = neighbor_lengths.max(axis=1)
    
    return features


def process_single_embedding(name: str, embedding_dir: Path, output_dir: Path, lengths: np.ndarray, k: int) -> np.ndarray:
    """Process a single embedding: load, build index, extract features."""
    print(f"\n{'='*60}\nProcessing: {name}\n{'='*60}")
    
    train_emb = load_embedding(embedding_dir, name, "train")
    print(f"  Train: {train_emb.shape}")
    
    test_path = embedding_dir / f"{name}_test.npy"
    has_test = test_path.exists()
    if has_test:
        test_emb = load_embedding(embedding_dir, name, "test")
        print(f"  Test: {test_emb.shape}")
    
    # Build index
    train_norm = train_emb.astype(np.float32).copy()
    faiss.normalize_L2(train_norm)
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_norm)
    print(f"  Index: {index.ntotal:,} vectors")
    
    # Extract and save features
    train_features = extract_knn_features_for_split("train", train_norm, index, lengths, k, is_train=True)
    np.save(output_dir / f"knn_k{k}_{name}_train.npy", train_features)
    print(f"  Saved train")
    
    if has_test:
        test_features = extract_knn_features_for_split("test", test_emb, index, lengths, k, is_train=False)
        np.save(output_dir / f"knn_k{k}_{name}_test.npy", test_features)
        print(f"  Saved test")
    
    return train_features


def main():
    parser = argparse.ArgumentParser(description="Extract KNN features from multiple embeddings")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--embeddings", type=str, nargs="*", default=["minilm", "mpnet", "distiluse", "e5small"])
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings")
    parser.add_argument("--data-dir", type=str, default="data/total_sentence_data/total_sentence_data")
    parser.add_argument("--output-dir", type=str, default="data/knn_features")
    args = parser.parse_args()
    
    embedding_dir, data_dir, output_dir = Path(args.embedding_dir), Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}\nMulti-Embedding KNN Feature Extraction\n{'='*70}")
    print(f"Embeddings: {args.embeddings}\nK: {args.k}\nOutput: {output_dir}")
    
    # Load lengths
    lengths = pd.read_csv(data_dir / "total_sentence_train.csv")["PRODUCT_LENGTH"].values.astype(np.float32)
    print(f"\nLengths: {len(lengths):,} samples, range [{lengths.min():.1f}, {lengths.max():.1f}]")
    
    # Process each embedding
    all_features = {}
    for name in args.embeddings:
        try:
            all_features[name] = process_single_embedding(name, embedding_dir, output_dir, lengths, args.k)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping {name}: {e}")
    
    # Summary
    print(f"\n{'='*70}\nSummary\n{'='*70}")
    print(f"Extracted: {list(all_features.keys())}")
    for name, features in all_features.items():
        print(f"\n  {name}:")
        for i, fname in enumerate(["knn_mean", "knn_std", "knn_median", "knn_min", "knn_max"]):
            print(f"    {fname:12s}: mean={features[:,i].mean():8.2f}, std={features[:,i].std():8.2f}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()

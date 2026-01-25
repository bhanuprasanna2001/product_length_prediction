"""Extract KNN features from embeddings for retrieval-augmented prediction."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_embedding(embedding_dir: Path, model: str, split: str) -> np.ndarray:
    path = embedding_dir / f"{model}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    return np.load(path)


def extract_knn_features_batch(index, embeddings: np.ndarray, lengths: np.ndarray, k: int, batch_size: int = 50000) -> np.ndarray:
    """Extract KNN features [mean, std, median, min, max] in batches."""
    import faiss
    
    n = len(embeddings)
    features = np.zeros((n, 5), dtype=np.float32)
    
    for i in tqdm(range(0, n, batch_size), desc="Extracting KNN features"):
        end = min(i + batch_size, n)
        batch = embeddings[i:end].astype(np.float32).copy()
        faiss.normalize_L2(batch)
        
        _, indices = index.search(batch, k)
        neighbor_lengths = lengths[indices]
        
        features[i:end, 0] = neighbor_lengths.mean(axis=1)
        features[i:end, 1] = neighbor_lengths.std(axis=1)
        features[i:end, 2] = np.median(neighbor_lengths, axis=1)
        features[i:end, 3] = neighbor_lengths.min(axis=1)
        features[i:end, 4] = neighbor_lengths.max(axis=1)
    
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract KNN features")
    parser.add_argument("--embedding", type=str, default="minilm", choices=["minilm", "mpnet", "distiluse", "e5small"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--embedding-dir", type=str, default="data/embeddings")
    parser.add_argument("--data-dir", type=str, default="data/total_sentence_data/total_sentence_data")
    parser.add_argument("--output-dir", type=str, default="data/knn_features")
    args = parser.parse_args()
    
    import faiss
    
    embedding_dir, data_dir, output_dir = Path(args.embedding_dir), Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\nKNN Feature Extraction\n{'='*60}")
    print(f"Embedding: {args.embedding}, K: {args.k}")
    
    # Load data
    print("\nLoading embeddings...")
    train_embeddings = load_embedding(embedding_dir, args.embedding, "train")
    print(f"  Train: {train_embeddings.shape}")
    
    test_path = embedding_dir / f"{args.embedding}_test.npy"
    has_test = test_path.exists()
    if has_test:
        test_embeddings = load_embedding(embedding_dir, args.embedding, "test")
        print(f"  Test: {test_embeddings.shape}")
    
    print("\nLoading lengths...")
    train_lengths = pd.read_csv(data_dir / "total_sentence_train.csv")["PRODUCT_LENGTH"].values.astype(np.float32)
    print(f"  Samples: {len(train_lengths):,}, Range: [{train_lengths.min():.1f}, {train_lengths.max():.1f}]")
    
    # Build index
    print("\nBuilding FAISS index...")
    train_norm = train_embeddings.astype(np.float32).copy()
    faiss.normalize_L2(train_norm)
    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_norm)
    print(f"  Built: {index.ntotal:,} vectors")
    
    # Extract train features (exclude self with k+1)
    print(f"\nExtracting train KNN features...")
    train_features = np.zeros((len(train_embeddings), 5), dtype=np.float32)
    batch_size = 50000
    
    for i in tqdm(range(0, len(train_embeddings), batch_size), desc="Train KNN"):
        end = min(i + batch_size, len(train_embeddings))
        _, indices = index.search(train_norm[i:end], args.k + 1)
        neighbor_lengths = train_lengths[indices[:, 1:]]  # Skip self
        
        train_features[i:end, 0] = neighbor_lengths.mean(axis=1)
        train_features[i:end, 1] = neighbor_lengths.std(axis=1)
        train_features[i:end, 2] = np.median(neighbor_lengths, axis=1)
        train_features[i:end, 3] = neighbor_lengths.min(axis=1)
        train_features[i:end, 4] = neighbor_lengths.max(axis=1)
    
    np.save(output_dir / f"knn_k{args.k}_{args.embedding}_train.npy", train_features)
    print(f"Saved train features: {train_features.shape}")
    
    # Extract test features
    if has_test:
        print(f"\nExtracting test KNN features...")
        test_features = extract_knn_features_batch(index, test_embeddings, train_lengths, args.k)
        np.save(output_dir / f"knn_k{args.k}_{args.embedding}_test.npy", test_features)
        print(f"Saved test features: {test_features.shape}")
    
    # Stats
    print(f"\n{'='*60}\nFeature Statistics (Train)\n{'='*60}")
    for i, name in enumerate(["knn_mean", "knn_std", "knn_median", "knn_min", "knn_max"]):
        print(f"  {name:12s}: mean={train_features[:,i].mean():8.2f}, std={train_features[:,i].std():8.2f}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()

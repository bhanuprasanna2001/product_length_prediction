"""KNN feature generation using FAISS for retrieval-augmented prediction."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
import faiss


@dataclass
class KNNFeatures:
    """KNN-derived features: mean, std, median, min, max of neighbor lengths."""
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    min: np.ndarray
    max: np.ndarray
    
    def to_array(self) -> np.ndarray:
        return np.column_stack([self.mean, self.std, self.median, self.min, self.max])
    
    @property
    def dim(self) -> int:
        return 5


class KNNIndex:
    """FAISS-based KNN index for fast cosine similarity search."""
    
    def __init__(self, embeddings: np.ndarray, lengths: np.ndarray, use_approximate: bool = False, hnsw_m: int = 32):
        self.lengths = lengths.astype(np.float32)
        self.embedding_dim = embeddings.shape[1]
        self.n_samples = len(embeddings)
        
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        if use_approximate:
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, hnsw_m)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(embeddings)
        
    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Find K nearest neighbors, returning (distances, indices)."""
        queries = queries.astype(np.float32).copy()
        faiss.normalize_L2(queries)
        return self.index.search(queries, k)
    
    def get_neighbor_lengths(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """Get lengths of K nearest neighbors."""
        _, indices = self.search(queries, k)
        return self.lengths[indices]
    
    def get_knn_features(self, queries: np.ndarray, k: int = 10) -> KNNFeatures:
        """Compute KNN features (mean, std, median, min, max) from neighbor lengths."""
        lengths = self.get_neighbor_lengths(queries, k)
        return KNNFeatures(
            mean=lengths.mean(axis=1),
            std=lengths.std(axis=1),
            median=np.median(lengths, axis=1),
            min=lengths.min(axis=1),
            max=lengths.max(axis=1),
        )
    
    def save(self, path: Path):
        path = Path(path)
        faiss.write_index(self.index, str(path / "knn_index.faiss"))
        np.save(path / "knn_lengths.npy", self.lengths)
        np.save(path / "knn_meta.npy", np.array([self.embedding_dim]))
        
    @classmethod
    def load(cls, path: Path) -> "KNNIndex":
        path = Path(path)
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path / "knn_index.faiss"))
        obj.lengths = np.load(path / "knn_lengths.npy")
        obj.embedding_dim = int(np.load(path / "knn_meta.npy")[0])
        obj.n_samples = len(obj.lengths)
        return obj


def generate_knn_features(
    train_embeddings: np.ndarray,
    train_lengths: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 10,
    use_approximate: bool = False,
    batch_size: int = 10000,
) -> KNNFeatures:
    """Generate KNN features for query embeddings using training data."""
    print(f"Building KNN index ({len(train_embeddings):,} samples, {train_embeddings.shape[1]}d)...")
    index = KNNIndex(train_embeddings, train_lengths, use_approximate=use_approximate)
    
    n_queries = len(query_embeddings)
    all_features = []
    
    print(f"Generating KNN features for {n_queries:,} queries (k={k})...")
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        features = index.get_knn_features(query_embeddings[i:end], k=k)
        all_features.append(features.to_array())
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end:,}/{n_queries:,}")
    
    combined = np.vstack(all_features)
    return KNNFeatures(mean=combined[:, 0], std=combined[:, 1], median=combined[:, 2], min=combined[:, 3], max=combined[:, 4])


def precompute_knn_features(
    embedding_dir: Path,
    output_dir: Path,
    embedding_model: str = "minilm",
    k: int = 10,
    use_approximate: bool = True,
):
    """Precompute and save KNN features for training data."""
    import pandas as pd
    
    embedding_dir, output_dir = Path(embedding_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings = np.load(embedding_dir / f"{embedding_model}_train.npy")
    lengths = np.asarray(pd.read_parquet(embedding_dir / "metadata_train.parquet")["PRODUCT_LENGTH"].values)
    print(f"Loaded {len(embeddings):,} embeddings ({embedding_model})")
    
    index = KNNIndex(embeddings, lengths, use_approximate=use_approximate)
    index.save(output_dir)
    
    print("\nGenerating KNN features for training data...")
    train_features = index.get_knn_features(embeddings, k=k+1)  # k+1 to exclude self
    np.save(output_dir / "knn_features_train.npy", train_features.to_array())
    
    print(f"\nKNN features saved to {output_dir}, shape: {train_features.to_array().shape}")

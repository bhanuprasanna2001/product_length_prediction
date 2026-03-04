"""Embedding extraction and KNN feature generation utilities."""

from pathlib import Path

import numpy as np
import torch

from .config import EMBEDDING_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_embeddings(
    texts: list[str],
    model_key: str,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Extract normalized sentence embeddings from a pre-trained model.

    Uses SentenceTransformers for encoding. Automatically applies the
    ``query: `` prefix required by E5 models.

    Returns float16 embeddings for storage efficiency.
    """
    from sentence_transformers import SentenceTransformer

    if model_key not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(EMBEDDING_REGISTRY)}")

    model_name, dim = EMBEDDING_REGISTRY[model_key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_key} ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # E5 models require a query prefix
    if "e5" in model_key.lower():
        texts = [f"query: {t}" for t in texts]

    print(f"Encoding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = embeddings.astype(np.float16)
    print(f"Done: shape={result.shape}, dtype={result.dtype}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_embeddings(embeddings: np.ndarray, model_key: str, split: str, output_dir: Path) -> Path:
    """Save embeddings as a .npy file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_key}_{split}.npy"
    np.save(path, embeddings)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved: {path} ({size_mb:.1f} MB)")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# KNN feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_knn_features(
    index,
    embeddings: np.ndarray,
    lengths: np.ndarray,
    k: int,
    is_train: bool = False,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Extract KNN statistical features: [mean, std, median, min, max] of neighbor lengths.

    Args:
        index: FAISS index (IndexFlatIP, already built).
        embeddings: Query embeddings (will be L2-normalized).
        lengths: Training target lengths for neighbor lookup.
        k: Number of neighbors to retrieve.
        is_train: If True, search k+1 and skip self-match in first position.
        batch_size: Batch size for FAISS search.

    Returns:
        Array of shape (n_queries, 5) with [mean, std, median, min, max] features.
    """
    import faiss

    n = len(embeddings)
    features = np.zeros((n, 5), dtype=np.float32)
    actual_k = k + 1 if is_train else k
    start_col = 1 if is_train else 0

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = embeddings[i:end].astype(np.float32).copy()
        faiss.normalize_L2(batch)

        _, indices = index.search(batch, actual_k)
        neighbor_lengths = lengths[indices[:, start_col:]]

        features[i:end, 0] = neighbor_lengths.mean(axis=1)
        features[i:end, 1] = neighbor_lengths.std(axis=1)
        features[i:end, 2] = np.median(neighbor_lengths, axis=1)
        features[i:end, 3] = neighbor_lengths.min(axis=1)
        features[i:end, 4] = neighbor_lengths.max(axis=1)

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {end:,}/{n:,}")

    return features


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build a FAISS inner-product index from L2-normalized embeddings."""
    import faiss

    emb = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    print(f"Built FAISS index: {index.ntotal:,} vectors, {emb.shape[1]}d")
    return index

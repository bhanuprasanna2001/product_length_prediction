"""
Embedding Extraction Utilities
==============================
Functions for extracting and saving embeddings from multilingual models.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Model registry
# 4 Multilingual + 1 Strong English
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",      # 384d
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",       # 768d
    "distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v1",   # 512d
    "e5small": "intfloat/multilingual-e5-small",                                  # 384d
    "allmpnet": "sentence-transformers/all-mpnet-base-v2",                        # 768d
}


def extract_embeddings(
    texts: list[str],
    model_key: str,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Extract embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_key: Key from EMBEDDING_MODELS
        batch_size: Batch size for encoding
        
    Returns:
        Embeddings array (n_texts, dim) in float16
    """
    model_path = EMBEDDING_MODELS[model_key]
    print(f"Loading {model_key}: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    
    # E5 models require prefix for best performance
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
    
    # Convert to float16 to save storage
    embeddings = embeddings.astype(np.float16)
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    model_key: str,
    split: str,
    output_dir: Path,
) -> Path:
    """Save embeddings to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{model_key}_{split}.npy"
    np.save(filepath, embeddings)
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved: {filepath} ({size_mb:.1f} MB)")
    
    return filepath


def save_metadata(
    df: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> Path:
    """Save metadata (product_id, product_type, target) as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"metadata_{split}.parquet"
    
    if split == "train":
        cols = ["PRODUCT_ID", "PRODUCT_TYPE_ID", "PRODUCT_LENGTH"]
    else:
        cols = ["PRODUCT_ID", "PRODUCT_TYPE_ID"]
    
    df[cols].to_parquet(filepath, index=False)
    print(f"Saved metadata: {filepath}")
    
    return filepath

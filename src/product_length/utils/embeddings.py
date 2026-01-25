"""Embedding extraction utilities using Sentence Transformers."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

from ..constants import get_embedding_model_name
from . import get_device


def extract_embeddings(texts: list[str], model_key: str, batch_size: int = 128) -> np.ndarray:
    """Extract normalized embeddings for texts (returns float16)."""
    model_path = get_embedding_model_name(model_key)
    print(f"Loading {model_key}: {model_path}")
    
    device = str(get_device())
    print(f"Using device: {device}")
    model = SentenceTransformer(model_path, device=device)
    
    # E5 models require query prefix
    if "e5" in model_key.lower():
        texts = [f"query: {t}" for t in texts]
    
    print(f"Encoding {len(texts):,} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float16)
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return embeddings


def save_embeddings(embeddings: np.ndarray, model_key: str, split: str, output_dir: Path) -> Path:
    """Save embeddings to .npy file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{model_key}_{split}.npy"
    np.save(filepath, embeddings)
    print(f"Saved: {filepath} ({filepath.stat().st_size / (1024 * 1024):.1f} MB)")
    return filepath


def save_metadata(df: pd.DataFrame, split: str, output_dir: Path) -> Path:
    """Save metadata (product_id, product_type, target) as parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"metadata_{split}.parquet"
    cols = ["PRODUCT_ID", "PRODUCT_TYPE_ID", "PRODUCT_LENGTH"] if split == "train" else ["PRODUCT_ID", "PRODUCT_TYPE_ID"]
    df[cols].to_parquet(filepath, index=False)
    print(f"Saved metadata: {filepath}")
    return filepath

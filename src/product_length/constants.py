"""Centralized constants for the product length prediction pipeline."""

from typing import Final


# Numerical stability
EPSILON: Final[float] = 1e-6
MIN_PREDICTION: Final[float] = 0.1

# Model architecture
KNN_PROJECTION_DIM: Final[int] = 32
KNN_FEATURE_COUNT: Final[int] = 5  # mean, std, median, min, max

# Training
WEIGHTED_MAPE_SCALE: Final[float] = 100.0
WEIGHTED_MAPE_MIN_WEIGHT: Final[float] = 0.1
SCORE_SCALE: Final[float] = 100.0
DEFAULT_SAMPLE_SIZE: Final[int] = 100

# Embedding model registry (single source of truth)
EMBEDDING_MODELS: Final[dict[str, dict[str, str | int]]] = {
    "minilm": {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384},
    "mpnet": {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "dim": 768},
    "distiluse": {"name": "sentence-transformers/distiluse-base-multilingual-cased-v1", "dim": 512},
    "e5small": {"name": "intfloat/multilingual-e5-small", "dim": 384},
    "allmpnet": {"name": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
    "labse": {"name": "sentence-transformers/LaBSE", "dim": 768},
    "e5base": {"name": "intfloat/multilingual-e5-base", "dim": 768},
    "bge": {"name": "BAAI/bge-m3", "dim": 1024},
}


def get_embedding_model_name(key: str) -> str:
    """Get HuggingFace model identifier for an embedding key."""
    if key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding: '{key}'. Available: {list(EMBEDDING_MODELS)}")
    return str(EMBEDDING_MODELS[key]["name"])


def get_embedding_dim(key: str) -> int:
    """Get output dimensionality for an embedding key."""
    if key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding: '{key}'. Available: {list(EMBEDDING_MODELS)}")
    return int(EMBEDDING_MODELS[key]["dim"])


# Loss function type names
class LossType:
    """Supported loss function identifiers."""
    MAPE: Final[str] = "mape"
    WEIGHTED_MAPE: Final[str] = "weighted_mape"
    MSE: Final[str] = "mse"
    HUBER: Final[str] = "huber"
    LOG_COSH: Final[str] = "log_cosh"
    SMAPE: Final[str] = "smape"
    LOG_MAPE: Final[str] = "log_mape"
    RMSLE: Final[str] = "rmsle"

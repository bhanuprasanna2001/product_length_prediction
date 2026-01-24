"""Utility functions and metrics."""

from .metrics import mape, rmsle, evaluate_predictions

# Embedding constants (no heavy imports)
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "labse": "sentence-transformers/LaBSE",
    "e5": "intfloat/multilingual-e5-base",
    "bge": "BAAI/bge-m3",
}

__all__ = ["mape", "rmsle", "evaluate_predictions", "EMBEDDING_MODELS"]

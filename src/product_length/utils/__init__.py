"""Utilities: metrics, losses, logging, and embeddings."""

import torch

from .metrics import mape, rmsle, evaluate_predictions
from .losses import (
    mape_loss,
    weighted_mape_loss,
    log_cosh_loss,
    huber_loss,
    mse_loss,
    smape_loss,
    log_mape_loss,
    rmsle_loss,
    get_loss_fn,
    register_loss,
    compute_mape,
    compute_rmsle,
    compute_score,
)
from ..constants import EMBEDDING_MODELS, get_embedding_model_name, get_embedding_dim


def get_device() -> torch.device:
    """Get best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "mape", "rmsle", "evaluate_predictions",
    "mape_loss", "weighted_mape_loss", "log_cosh_loss", "huber_loss", "mse_loss",
    "smape_loss", "log_mape_loss", "rmsle_loss",
    "get_loss_fn", "register_loss", "compute_mape", "compute_rmsle", "compute_score",
    "EMBEDDING_MODELS", "get_embedding_model_name", "get_embedding_dim",
    "get_device",
]
"""PyTorch loss functions and evaluation metrics for product length prediction."""

import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1e-6

# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════


def mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Percentage Error — directly optimizes the competition metric."""
    return torch.mean(torch.abs(target - pred) / target)


def weighted_mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAPE weighted by target magnitude to reduce influence of small values."""
    weights = torch.clamp(target / 100.0, min=0.1, max=1.0)
    return torch.mean(weights * torch.abs(target - pred) / target)


def smape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric MAPE — bounded [0, 2], handles near-zero values gracefully."""
    return torch.mean(2.0 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + EPSILON))


def log_mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAPE in log-space — better gradient flow for wide value ranges."""
    log_pred = torch.log(pred + EPSILON)
    log_target = torch.log(target + EPSILON)
    return torch.mean(torch.abs(log_pred - log_target) / (torch.abs(log_target) + EPSILON))


def rmsle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Log Error — penalizes under-prediction more."""
    return torch.sqrt(torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2))


def focal_mape_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal MAPE — upweights hard examples with high error."""
    ape = torch.abs(target - pred) / target
    weight = (ape ** gamma) / (ape ** gamma).mean().clamp(min=EPSILON)
    return torch.mean(weight * ape)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """Weighted combination of MAPE and RMSLE."""
    return alpha * mape_loss(pred, target) + (1 - alpha) * rmsle_loss(pred, target)


# ── Loss registry ──────────────────────────────────────────────────────────────

_REGISTRY = {
    "mape": mape_loss,
    "weighted_mape": weighted_mape_loss,
    "smape": smape_loss,
    "log_mape": log_mape_loss,
    "rmsle": rmsle_loss,
    "focal_mape": focal_mape_loss,
    "combined": combined_loss,
}


def get_loss_fn(name: str):
    """Get loss function by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (PyTorch — used in training step)
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAPE as percentage (0–100+) for logging."""
    return torch.mean(torch.abs(target - pred) / target) * 100


@torch.no_grad()
def compute_rmsle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Log Error."""
    return torch.sqrt(torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2))


@torch.no_grad()
def compute_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Competition score: max(0, 100 × (1 − MAPE/100))."""
    return torch.clamp(100.0 - compute_mape(pred, target), min=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (NumPy — used in evaluation)
# ═══════════════════════════════════════════════════════════════════════════════


def mape_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE as percentage."""
    return float(np.mean(np.abs((y_true - np.maximum(y_pred, EPSILON)) / y_true)) * 100)


def rmsle_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error."""
    return float(np.sqrt(np.mean((np.log1p(np.maximum(y_pred, EPSILON)) - np.log1p(y_true)) ** 2)))


def score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition score."""
    return max(0.0, 100.0 * (1 - mape_numpy(y_true, y_pred) / 100))

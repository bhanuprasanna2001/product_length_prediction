"""PyTorch loss functions for product length prediction."""

import torch
import torch.nn.functional as F
from typing import Callable

from ..constants import EPSILON, SCORE_SCALE, WEIGHTED_MAPE_MIN_WEIGHT, WEIGHTED_MAPE_SCALE, LossType

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAPE loss (directly optimizes competition metric)."""
    pred_safe = F.relu(pred) + EPSILON
    return torch.mean(torch.abs(target - pred_safe) / target)


def weighted_mape_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    min_weight: float = WEIGHTED_MAPE_MIN_WEIGHT,
    scale: float = WEIGHTED_MAPE_SCALE,
) -> torch.Tensor:
    """Weighted MAPE that downweights small targets."""
    pred_safe = F.relu(pred) + EPSILON
    weights = torch.clamp(target / scale, min=min_weight, max=1.0)
    return torch.mean(weights * torch.abs(target - pred_safe) / target)


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-Cosh loss in log space (smooth MAE approximation)."""
    pred_safe = F.relu(pred) + EPSILON
    diff = torch.log1p(pred_safe) - torch.log1p(target)
    return torch.mean(torch.log(torch.cosh(diff)))


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss in log space."""
    pred_safe = F.relu(pred) + EPSILON
    return F.huber_loss(torch.log1p(pred_safe), torch.log1p(target), delta=delta)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss in log space."""
    pred_safe = F.relu(pred) + EPSILON
    return F.mse_loss(torch.log1p(pred_safe), torch.log1p(target))


def smape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric MAPE loss (bounded 0-2, handles zeros gracefully)."""
    pred_safe = F.relu(pred) + EPSILON
    numerator = torch.abs(pred_safe - target)
    denominator = torch.abs(pred_safe) + torch.abs(target) + EPSILON
    return torch.mean(2.0 * numerator / denominator)


def log_mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-MAPE loss (better for wide value ranges)."""
    pred_safe = F.relu(pred) + EPSILON
    log_pred, log_target = torch.log(pred_safe), torch.log(target + EPSILON)
    return torch.mean(torch.abs(log_pred - log_target) / (torch.abs(log_target) + EPSILON))


def rmsle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSLE loss (penalizes under-prediction more)."""
    pred_safe = F.relu(pred) + EPSILON
    return torch.sqrt(torch.mean((torch.log1p(pred_safe) - torch.log1p(target)) ** 2))


# Loss registry
_LOSS_REGISTRY: dict[str, Callable[..., LossFn]] = {
    LossType.MAPE: lambda **_: mape_loss,
    LossType.WEIGHTED_MAPE: lambda **kw: lambda p, t: weighted_mape_loss(p, t, **kw),
    LossType.MSE: lambda **_: mse_loss,
    LossType.HUBER: lambda delta=1.0, **_: lambda p, t: huber_loss(p, t, delta=delta),
    LossType.LOG_COSH: lambda **_: log_cosh_loss,
    LossType.SMAPE: lambda **_: smape_loss,
    LossType.LOG_MAPE: lambda **_: log_mape_loss,
    LossType.RMSLE: lambda **_: rmsle_loss,
}


def get_loss_fn(name: str, **kwargs) -> LossFn:
    """Get loss function by name."""
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {', '.join(_LOSS_REGISTRY.keys())}")
    return _LOSS_REGISTRY[name](**kwargs)


def register_loss(name: str, factory: Callable[..., LossFn]) -> None:
    """Register a custom loss function."""
    if name in _LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' already registered")
    _LOSS_REGISTRY[name] = factory


# PyTorch metrics for training step logging
def compute_mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAPE as percentage (0-100+)."""
    with torch.no_grad():
        pred_safe = F.relu(pred) + EPSILON
        return torch.mean(torch.abs((target - pred_safe) / target)) * SCORE_SCALE


def compute_rmsle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Logarithmic Error."""
    with torch.no_grad():
        pred_safe = F.relu(pred) + EPSILON
        return torch.sqrt(torch.mean((torch.log1p(pred_safe) - torch.log1p(target)) ** 2))


def compute_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Competition score: max(0, 100 * (1 - MAPE/100))."""
    return torch.clamp(SCORE_SCALE - compute_mape(pred, target), min=0)

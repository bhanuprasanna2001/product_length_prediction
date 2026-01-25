"""Protocol definitions for type-safe dependency injection."""

from typing import Protocol, runtime_checkable
import numpy as np
import torch
from numpy.typing import NDArray


@runtime_checkable
class LossFunction(Protocol):
    """Callable that computes loss between predictions and targets."""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class Snapper(Protocol):
    """Maps predictions to discrete valid values."""
    
    def snap(self, predictions: NDArray[np.float64], valid_values: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extracts fixed-size features from input data."""
    
    def extract(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        ...
    
    @property
    def output_dim(self) -> int:
        ...


# Type aliases for array shapes
EmbeddingArray = NDArray[np.float32]  # (n_samples, embedding_dim)
LengthArray = NDArray[np.float64]      # (n_samples,)
ProductTypeArray = NDArray[np.int64]   # (n_samples,)
BatchDict = dict[str, torch.Tensor]
MetricsDict = dict[str, float]

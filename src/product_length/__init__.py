"""Product length prediction from text using multilingual embeddings.

Predicts product dimensions from text metadata (title, description, bullets)
using pre-computed multilingual embeddings and MLP regression optimized for MAPE.
"""

__version__ = "2.0.0"

from .config import Config, ConfigurationError
from .constants import EPSILON, EMBEDDING_MODELS, LossType, get_embedding_dim, get_embedding_model_name
from .protocols import LossFunction, Snapper, FeatureExtractor

__all__ = [
    "__version__",
    "Config",
    "ConfigurationError",
    "EPSILON",
    "EMBEDDING_MODELS",
    "LossType",
    "get_embedding_dim",
    "get_embedding_model_name",
    "LossFunction",
    "Snapper",
    "FeatureExtractor",
]
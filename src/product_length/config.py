"""
Configuration Management
========================
Loads and validates YAML configuration files.

Follows clean code principles:
- Dataclasses for type safety
- Validation on load
- Clear error messages
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


@dataclass
class DataConfig:
    """Data paths and train/val/test split ratios."""
    train_path: Path
    test_path: Path
    embedding_dir: Path
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    
    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ConfigurationError(f"Split ratios must sum to 1.0, got {total:.3f}")


@dataclass
class EmbeddingModelConfig:
    """Single embedding model specification."""
    name: str
    dim: int


@dataclass 
class EmbeddingsConfig:
    """Multi-embedding ensemble configuration."""
    models: dict[str, EmbeddingModelConfig]
    active: list[str]
    batch_size: int = 128
    
    def __post_init__(self) -> None:
        undefined = set(self.active) - set(self.models.keys())
        if undefined:
            raise ConfigurationError(f"Unknown embeddings: {undefined}. Available: {list(self.models)}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive: {self.batch_size}")
    
    @property
    def total_dim(self) -> int:
        """Combined dimensionality of active embeddings."""
        return sum(self.models[m].dim for m in self.active)


@dataclass
class ModelConfig:
    """MLP architecture hyperparameters."""
    product_type_emb_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 256, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    def __post_init__(self) -> None:
        if self.product_type_emb_dim <= 0:
            raise ConfigurationError(f"product_type_emb_dim must be positive: {self.product_type_emb_dim}")
        if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
            raise ConfigurationError(f"hidden_dims must be non-empty positive integers: {self.hidden_dims}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigurationError(f"dropout must be in [0, 1): {self.dropout}")


@dataclass
class TrainingConfig:
    """Optimization and training loop hyperparameters."""
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_ratio: float = 0.05
    loss_fn: str = "mape"
    huber_delta: float = 0.5
    patience: int = 5
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"
    
    VALID_LOSSES: tuple[str, ...] = (
        "mse", "mae", "huber", "mape", "weighted_mape", "log_cosh", "smape", "log_mape", "rmsle"
    )
    
    def __post_init__(self) -> None:
        errors = []
        if self.batch_size <= 0:
            errors.append(f"batch_size={self.batch_size}")
        if self.lr <= 0:
            errors.append(f"lr={self.lr}")
        if self.epochs <= 0:
            errors.append(f"epochs={self.epochs}")
        if self.patience <= 0:
            errors.append(f"patience={self.patience}")
        if self.weight_decay < 0:
            errors.append(f"weight_decay={self.weight_decay}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            errors.append(f"warmup_ratio={self.warmup_ratio}")
        if self.loss_fn not in self.VALID_LOSSES:
            errors.append(f"loss_fn='{self.loss_fn}' not in {self.VALID_LOSSES}")
        if errors:
            raise ConfigurationError(f"Invalid training config: {', '.join(errors)}")


@dataclass
class PostprocessingConfig:
    """Prediction snapping configuration."""
    use_snapping: bool = True
    snap_by_type: bool = True


@dataclass
class LoggingConfig:
    """Experiment tracking configuration."""
    project: str = "amazon-product-length"
    run_name: str | None = None
    log_every_n_steps: int = 50
    val_check_interval: float = 0.25


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""
    dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_top_k: int = 3
    monitor: str = "val_mape"
    mode: str = "min"


@dataclass
class FeaturesConfig:
    """Feature engineering options (KNN retrieval, log-transform)."""
    use_knn: bool = False
    knn_k: int = 20
    knn_embeddings: list[str] = field(default_factory=lambda: ["minilm"])
    knn_ensemble: str = "concat"
    use_log_target: bool = False
    
    VALID_ENSEMBLES: tuple[str, ...] = ("concat", "mean", "weighted")
    
    def __post_init__(self) -> None:
        if self.knn_k <= 0 or self.knn_k > 100:
            raise ConfigurationError(f"knn_k must be in (0, 100]: {self.knn_k}")
        if not self.knn_embeddings:
            raise ConfigurationError("knn_embeddings cannot be empty")
        if self.knn_ensemble not in self.VALID_ENSEMBLES:
            raise ConfigurationError(f"knn_ensemble='{self.knn_ensemble}' not in {self.VALID_ENSEMBLES}")
    
    @property
    def knn_embedding(self) -> str:
        """Primary KNN embedding (for backwards compatibility)."""
        return self.knn_embeddings[0]


@dataclass
class SystemConfig:
    """Runtime configuration."""
    seed: int = 42
    num_workers: int = 4


@dataclass
class Config:
    """Root configuration container."""
    data: DataConfig
    embeddings: EmbeddingsConfig
    model: ModelConfig
    training: TrainingConfig
    postprocessing: PostprocessingConfig
    logging: LoggingConfig
    checkpoints: CheckpointConfig
    features: FeaturesConfig
    system: SystemConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}")
        
        if raw is None:
            raise ConfigurationError(f"Empty configuration file: {path}")
            
        return cls.from_dict(raw)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create and validate Config from dictionary."""
        required = {"data", "embeddings", "model", "training", "postprocessing", "logging", "checkpoints", "system"}
        missing = required - set(d.keys())
        if missing:
            raise ConfigurationError(f"Missing sections: {missing}")
        
        try:
            emb_models = {k: EmbeddingModelConfig(**v) for k, v in d["embeddings"]["models"].items()}
            
            # Features with backwards compatibility for knn_embedding -> knn_embeddings
            features_dict = d.get("features", {})
            knn_embeddings = features_dict.get("knn_embeddings")
            if knn_embeddings is None:
                old = features_dict.get("knn_embedding", "minilm")
                knn_embeddings = [old] if isinstance(old, str) else old
            
            return cls(
                data=DataConfig(
                    train_path=Path(d["data"]["train_path"]),
                    test_path=Path(d["data"]["test_path"]),
                    embedding_dir=Path(d["data"]["embedding_dir"]),
                    train_ratio=d["data"]["train_ratio"],
                    val_ratio=d["data"]["val_ratio"],
                    test_ratio=d["data"]["test_ratio"],
                ),
                embeddings=EmbeddingsConfig(
                    models=emb_models,
                    active=d["embeddings"]["active"],
                    batch_size=d["embeddings"]["batch_size"],
                ),
                model=ModelConfig(**d["model"]),
                training=TrainingConfig(**d["training"]),
                postprocessing=PostprocessingConfig(**d["postprocessing"]),
                logging=LoggingConfig(**d["logging"]),
                checkpoints=CheckpointConfig(
                    dir=Path(d["checkpoints"]["dir"]),
                    save_top_k=d["checkpoints"]["save_top_k"],
                    monitor=d["checkpoints"]["monitor"],
                    mode=d["checkpoints"]["mode"],
                ),
                features=FeaturesConfig(
                    use_knn=features_dict.get("use_knn", False),
                    knn_k=features_dict.get("knn_k", 20),
                    knn_embeddings=knn_embeddings,
                    knn_ensemble=features_dict.get("knn_ensemble", "concat"),
                    use_log_target=features_dict.get("use_log_target", False),
                ),
                system=SystemConfig(**d["system"]),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key: {e}")
        except TypeError as e:
            raise ConfigurationError(f"Invalid value: {e}")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging."""
        return {
            "data": {"train_ratio": self.data.train_ratio, "val_ratio": self.data.val_ratio},
            "embeddings": {"active": self.embeddings.active, "total_dim": self.embeddings.total_dim},
            "model": {"hidden_dims": self.model.hidden_dims, "dropout": self.model.dropout},
            "training": {"batch_size": self.training.batch_size, "lr": self.training.lr, "loss_fn": self.training.loss_fn},
            "features": {"use_knn": self.features.use_knn, "knn_k": self.features.knn_k},
        }


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)

"""
Configuration Management
========================
Loads and validates YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    train_path: Path
    test_path: Path
    embedding_dir: Path
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05


@dataclass
class EmbeddingModelConfig:
    name: str
    dim: int


@dataclass 
class EmbeddingsConfig:
    models: dict[str, EmbeddingModelConfig]
    active: list[str]
    batch_size: int = 128
    
    @property
    def active_models(self) -> list[str]:
        return self.active
    
    @property
    def total_dim(self) -> int:
        return sum(self.models[m].dim for m in self.active)


@dataclass
class ModelConfig:
    product_type_emb_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 256, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_ratio: float = 0.05
    loss_fn: str = "huber"
    huber_delta: float = 0.5
    patience: int = 5
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"


@dataclass
class PostprocessingConfig:
    use_calibration: bool = True
    calibration_method: str = "isotonic"
    use_snapping: bool = True
    snap_by_type: bool = True


@dataclass
class LoggingConfig:
    project: str = "amazon-product-length"
    run_name: str | None = None
    log_every_n_steps: int = 50
    val_check_interval: float = 0.25


@dataclass
class CheckpointConfig:
    dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_top_k: int = 3
    monitor: str = "val_mape"
    mode: str = "min"


@dataclass
class SystemConfig:
    seed: int = 42
    num_workers: int = 4


@dataclass
class Config:
    """Master configuration class."""
    data: DataConfig
    embeddings: EmbeddingsConfig
    model: ModelConfig
    training: TrainingConfig
    postprocessing: PostprocessingConfig
    logging: LoggingConfig
    checkpoints: CheckpointConfig
    system: SystemConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Parse embedding models
        emb_models = {
            k: EmbeddingModelConfig(**v) 
            for k, v in d["embeddings"]["models"].items()
        }
        
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
            system=SystemConfig(**d["system"]),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for W&B logging)."""
        return {
            "data": {
                "train_path": str(self.data.train_path),
                "test_path": str(self.data.test_path),
                "embedding_dir": str(self.data.embedding_dir),
                "train_ratio": self.data.train_ratio,
                "val_ratio": self.data.val_ratio,
                "test_ratio": self.data.test_ratio,
            },
            "embeddings": {
                "active": self.embeddings.active,
                "total_dim": self.embeddings.total_dim,
            },
            "model": {
                "product_type_emb_dim": self.model.product_type_emb_dim,
                "hidden_dims": self.model.hidden_dims,
                "dropout": self.model.dropout,
                "use_batch_norm": self.model.use_batch_norm,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "lr": self.training.lr,
                "weight_decay": self.training.weight_decay,
                "epochs": self.training.epochs,
                "loss_fn": self.training.loss_fn,
            },
        }


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)

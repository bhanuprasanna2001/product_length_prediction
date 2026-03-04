"""Configuration for product length prediction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Embedding model registry: short key → (HuggingFace ID, output dim)
EMBEDDING_REGISTRY: dict[str, tuple[str, int]] = {
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 768),
    "distiluse": ("sentence-transformers/distiluse-base-multilingual-cased-v1", 512),
    "e5small": ("intfloat/multilingual-e5-small", 384),
    "allmpnet": ("sentence-transformers/all-mpnet-base-v2", 768),
    "labse": ("sentence-transformers/LaBSE", 768),
    "e5base": ("intfloat/multilingual-e5-base", 768),
}


@dataclass
class Config:
    """Single flat configuration for the entire pipeline.

    Intentionally kept as one class — this project is small enough
    that nested config hierarchies add complexity without value.
    """

    # ── Data paths ──────────────────────────────────────────────
    data_dir: Path = Path("data/total_sentence_data/total_sentence_data")
    embedding_dir: Path = Path("data/embeddings")
    knn_dir: Path = Path("data/knn_features")
    output_dir: Path = Path("checkpoints")

    # ── Data splits ─────────────────────────────────────────────
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05

    # ── Embeddings ──────────────────────────────────────────────
    active_embeddings: list[str] = field(
        default_factory=lambda: ["minilm", "mpnet", "distiluse", "e5small"]
    )

    # ── KNN features ────────────────────────────────────────────
    use_knn: bool = True
    knn_k: int = 20
    knn_embeddings: list[str] = field(
        default_factory=lambda: ["minilm", "mpnet", "distiluse", "e5small"]
    )

    # ── Model architecture ──────────────────────────────────────
    product_type_emb_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 256, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True

    # ── Training ────────────────────────────────────────────────
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_ratio: float = 0.05
    patience: int = 5
    gradient_clip_val: float = 1.0
    loss_fn: str = "mape"
    use_log_target: bool = True

    # ── Logging ─────────────────────────────────────────────────
    wandb_project: str = "amazon-product-length"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 50
    val_check_interval: float = 0.25

    # ── System ──────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 2

    # ── Derived properties ──────────────────────────────────────

    @property
    def embedding_dims(self) -> dict[str, int]:
        """Dimension of each active embedding model."""
        return {key: EMBEDDING_REGISTRY[key][1] for key in self.active_embeddings}

    @property
    def total_embedding_dim(self) -> int:
        """Combined dimensionality of all active embeddings."""
        return sum(self.embedding_dims.values())

    @property
    def knn_dim(self) -> int:
        """Dimensionality of concatenated KNN feature vector."""
        if not self.use_knn:
            return 0
        n_stats = 5  # mean, std, median, min, max
        return n_stats * len(self.knn_embeddings)

    # ── Serialization ───────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Empty config file: {path}")
        return cls(**{k: Path(v) if k.endswith("_dir") else v for k, v in raw.items()})

    def to_dict(self) -> dict[str, Any]:
        """Serialize for W&B logging."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        } | {
            "total_embedding_dim": self.total_embedding_dim,
            "knn_dim": self.knn_dim,
        }

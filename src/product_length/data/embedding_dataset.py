"""PyTorch Dataset and Lightning DataModule for pre-computed embeddings.

Memory-efficient implementation using memory-mapped arrays. Indices are stored
per split; actual embedding data is accessed on-demand from shared mmap files.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..constants import KNN_FEATURE_COUNT

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings with optional KNN features.
    
    Stores indices into memory-mapped files rather than copying data.
    Embedding lookup happens in __getitem__ for memory efficiency.
    """
    
    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        indices: np.ndarray,
        product_types: np.ndarray,
        targets: Optional[np.ndarray] = None,
        product_ids: Optional[np.ndarray] = None,
        knn_features: Optional[dict[str, np.ndarray] | np.ndarray] = None,
        knn_ensemble: str = "concat",
        use_log_target: bool = False,
    ):
        self.embeddings = embeddings
        self.indices = indices
        self.product_types = product_types
        self.targets = targets
        self.product_ids = product_ids
        self.knn_features = knn_features
        self.knn_ensemble = knn_ensemble
        self.use_log_target = use_log_target
        self.model_names = list(embeddings.keys())
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        
        # Concatenate embeddings from all active models
        emb_list = [self.embeddings[name][real_idx] for name in self.model_names]
        text_embedding = np.concatenate(emb_list, axis=0)
        
        item = {
            "text_embedding": torch.tensor(text_embedding, dtype=torch.float32),
            "product_type": torch.tensor(self.product_types[idx], dtype=torch.long),
        }
        
        if self.targets is not None:
            target = self.targets[idx]
            if self.use_log_target:
                target = np.log1p(target)
            item["target"] = torch.tensor(target, dtype=torch.float32)
            
        if self.product_ids is not None:
            item["product_id"] = self.product_ids[idx]
        
        if self.knn_features is not None:
            item["knn_features"] = self._get_knn_features(real_idx)
            
        return item
    
    def _get_knn_features(self, real_idx: int) -> torch.Tensor:
        """Combine KNN features from single or multiple embedding sources."""
        if self.knn_features is None:
            return torch.tensor([], dtype=torch.float32)
        if isinstance(self.knn_features, dict):
            knn_list = [self.knn_features[name][real_idx] for name in sorted(self.knn_features.keys())]
            if self.knn_ensemble == "mean":
                combined = np.mean(knn_list, axis=0)
            else:  # concat (default)
                combined = np.concatenate(knn_list, axis=0)
            return torch.tensor(combined, dtype=torch.float32)
        return torch.tensor(self.knn_features[real_idx], dtype=torch.float32)


class DataLoaderFactory:
    """Creates DataLoaders with consistent settings."""
    
    @staticmethod
    def create(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        # pin_memory only works with CUDA, not MPS
        use_pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=num_workers > 0,
        )


class EmbeddingDataModule(pl.LightningDataModule):
    """Lightning DataModule for embedding-based training.
    
    Handles loading embeddings, KNN features, product type mapping,
    and train/val/test split creation.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding_dir = Path(config.data.embedding_dir)
        self.active_models = config.embeddings.active
        
        # KNN configuration
        self.use_knn = config.features.use_knn
        self.knn_k = config.features.knn_k
        self.knn_embeddings = config.features.knn_embeddings
        self.knn_ensemble = config.features.knn_ensemble
        self.use_log_target = config.features.use_log_target
        
        # Compute KNN feature dimension
        if self.use_knn:
            n_sources = len(self.knn_embeddings)
            self.knn_dim = KNN_FEATURE_COUNT * n_sources if self.knn_ensemble == "concat" else KNN_FEATURE_COUNT
        else:
            self.knn_dim = 0
        
        # Populated in setup()
        self.train_ds: Optional[EmbeddingDataset] = None
        self.val_ds: Optional[EmbeddingDataset] = None
        self.test_ds: Optional[EmbeddingDataset] = None
        self.product_type_map: Optional[dict[int, int]] = None
        self.num_product_types: int = 0
        self.train_lengths_by_type: dict[int, np.ndarray] = {}
        self.all_train_lengths: Optional[np.ndarray] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create train/val/test datasets."""
        metadata = self._load_metadata("train")
        n_samples = len(metadata)
        
        self._build_product_type_map(metadata)
        assert self.product_type_map is not None
        
        product_types = np.array(metadata["PRODUCT_TYPE_ID"].map(self.product_type_map).values)
        targets = np.array(metadata["PRODUCT_LENGTH"].values)
        product_ids = np.array(metadata["PRODUCT_ID"].values)
        
        embeddings = self._load_embeddings("train")
        knn_features = self._load_knn_features("train") if self.use_knn else None
        
        train_idx, val_idx, test_idx = self._create_splits(n_samples)
        self._log_split_info(train_idx, val_idx, test_idx)
        
        self.train_ds = self._create_dataset(embeddings, product_types, targets, product_ids, knn_features, train_idx)
        self.val_ds = self._create_dataset(embeddings, product_types, targets, product_ids, knn_features, val_idx)
        self.test_ds = self._create_dataset(embeddings, product_types, targets, product_ids, knn_features, test_idx)
        
        self._compute_length_statistics(targets[train_idx], product_types[train_idx])
    
    def _load_metadata(self, split: str) -> pd.DataFrame:
        path = self.embedding_dir / f"metadata_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")
        return pd.read_parquet(path)
    
    def _build_product_type_map(self, metadata: pd.DataFrame) -> None:
        """Create product_type_id -> index mapping (0 reserved for unknown)."""
        unique_types = metadata["PRODUCT_TYPE_ID"].unique()
        self.product_type_map = {t: i + 1 for i, t in enumerate(sorted(unique_types))}
        self.num_product_types = len(unique_types) + 1
        
    def _load_embeddings(self, split: str) -> dict[str, np.ndarray]:
        """Load memory-mapped embeddings for all active models."""
        embeddings = {}
        for model in self.active_models:
            path = self.embedding_dir / f"{model}_{split}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}. Run scripts/extract_embeddings.py --split {split}")
            embeddings[model] = np.load(path, mmap_mode="r")
            logger.info(f"Loaded {model}: {embeddings[model].shape}")
        return embeddings
    
    def _load_knn_features(self, split: str) -> dict[str, np.ndarray] | np.ndarray:
        """Load pre-computed KNN features (memory-mapped)."""
        knn_dir = self.embedding_dir.parent / "knn_features"
        
        if len(self.knn_embeddings) == 1:
            emb_name = self.knn_embeddings[0]
            path = knn_dir / f"knn_k{self.knn_k}_{emb_name}_{split}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}. Run scripts/extract_knn_features.py --k {self.knn_k} --embedding {emb_name}")
            knn = np.load(path, mmap_mode="r")
            logger.info(f"Loaded KNN ({emb_name}): {knn.shape}")
            return knn
        
        # Multi-embedding KNN
        knn_dict = {}
        missing = []
        for emb_name in self.knn_embeddings:
            path = knn_dir / f"knn_k{self.knn_k}_{emb_name}_{split}.npy"
            if not path.exists():
                missing.append(emb_name)
            else:
                knn_dict[emb_name] = np.load(path, mmap_mode="r")
                logger.info(f"Loaded KNN ({emb_name}): {knn_dict[emb_name].shape}")
        
        if missing:
            raise FileNotFoundError(f"Missing KNN for: {missing}. Run scripts/extract_knn_features.py --k {self.knn_k}")
        
        logger.info(f"Multi-KNN ({self.knn_ensemble}): {len(knn_dict)} sources")
        return knn_dict
    
    def _create_splits(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create reproducible train/val/test index splits."""
        rng = np.random.default_rng(self.config.system.seed)
        indices = rng.permutation(n_samples)
        train_end = int(self.config.data.train_ratio * n_samples)
        val_end = int((self.config.data.train_ratio + self.config.data.val_ratio) * n_samples)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]
    
    def _create_dataset(
        self,
        embeddings: dict[str, np.ndarray],
        product_types: np.ndarray,
        targets: np.ndarray,
        product_ids: np.ndarray,
        knn_features: Optional[dict[str, np.ndarray] | np.ndarray],
        indices: np.ndarray,
    ) -> EmbeddingDataset:
        """Create dataset for a split. Passes full mmap arrays to avoid copying."""
        return EmbeddingDataset(
            embeddings=embeddings,
            indices=indices,
            product_types=product_types[indices],
            targets=targets[indices],
            product_ids=product_ids[indices],
            knn_features=knn_features,
            knn_ensemble=self.knn_ensemble,
            use_log_target=self.use_log_target,
        )
    
    def _compute_length_statistics(self, targets: np.ndarray, product_types: np.ndarray) -> None:
        """Cache valid lengths for snapping post-processor."""
        self.all_train_lengths = np.unique(targets)
        self.train_lengths_by_type = {
            int(ptype): np.unique(targets[product_types == ptype])
            for ptype in np.unique(product_types)
        }
        logger.info(f"Unique lengths: {len(self.all_train_lengths):,}, product types: {len(self.train_lengths_by_type):,}")
    
    def _log_split_info(self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
        logger.info(f"Splits: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
        if self.use_knn:
            logger.info(f"KNN: k={self.knn_k}, embeddings={self.knn_embeddings}, ensemble={self.knn_ensemble}")
        if self.use_log_target:
            logger.info("Log-target transform enabled")
    
    def get_inference_dataset(self) -> EmbeddingDataset:
        """Create dataset for competition test set (no targets)."""
        metadata = self._load_metadata("test")
        if self.product_type_map is None:
            raise RuntimeError("setup() must be called before get_inference_dataset()")
        product_type_map = self.product_type_map  # Type narrowing for lambda
        product_types = metadata["PRODUCT_TYPE_ID"].map(lambda x: product_type_map.get(x, 0)).values
        product_ids = metadata["PRODUCT_ID"].values
        
        return EmbeddingDataset(
            embeddings=self._load_embeddings("test"),
            indices=np.arange(len(metadata)),
            product_types=np.array(product_types),
            product_ids=np.array(product_ids),
            knn_features=self._load_knn_features("test") if self.use_knn else None,
            knn_ensemble=self.knn_ensemble,
            use_log_target=False,
        )
    
    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoaderFactory.create(self.train_ds, self.config.training.batch_size, shuffle=True, num_workers=self.config.system.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoaderFactory.create(self.val_ds, self.config.training.batch_size * 2, shuffle=False, num_workers=self.config.system.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoaderFactory.create(self.test_ds, self.config.training.batch_size * 2, shuffle=False, num_workers=self.config.system.num_workers)

"""
Embedding Dataset Module
========================
PyTorch Dataset and DataModule for loading pre-computed embeddings.
Optimized for fast training with memory-mapped numpy arrays.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path

from ..config import Config


class EmbeddingDataset(Dataset):
    """
    Dataset for pre-computed embeddings.
    
    Memory-maps large embedding files to avoid loading everything into RAM.
    Concatenates embeddings from multiple models on-the-fly.
    """
    
    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        product_types: np.ndarray,
        targets: np.ndarray | None = None,
        product_ids: np.ndarray | None = None,
    ):
        self.embeddings = embeddings
        self.product_types = product_types
        self.targets = targets
        self.product_ids = product_ids
        self.model_names = list(embeddings.keys())
        
    def __len__(self) -> int:
        return len(self.product_types)
    
    def __getitem__(self, idx: int) -> dict:
        # Concatenate embeddings from all models
        emb_list = [self.embeddings[name][idx] for name in self.model_names]
        text_embedding = np.concatenate(emb_list, axis=0)
        
        item = {
            "text_embedding": torch.tensor(text_embedding, dtype=torch.float32),
            "product_type": torch.tensor(self.product_types[idx], dtype=torch.long),
        }
        
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            
        if self.product_ids is not None:
            item["product_id"] = self.product_ids[idx]
            
        return item


class EmbeddingDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for embedding-based training.
    
    Handles:
    - Loading pre-computed embeddings (memory-mapped)
    - Building product type mapping
    - Train/val/test splits
    - Computing statistics for post-processing
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding_dir = config.data.embedding_dir
        self.active_models = config.embeddings.active
        
        # Populated in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.product_type_map = None
        self.num_product_types = None
        self.train_lengths_by_type = None
        self.all_train_lengths = None
        
    def setup(self, stage: str | None = None):
        """Load embeddings and create datasets."""
        metadata_path = self.embedding_dir / "metadata_train.parquet"
        metadata = pd.read_parquet(metadata_path)
        n_samples = len(metadata)
        
        # Build product type mapping
        unique_types = metadata["PRODUCT_TYPE_ID"].unique()
        self.product_type_map = {t: i + 1 for i, t in enumerate(sorted(unique_types))}
        self.num_product_types = len(unique_types) + 1
        
        # Map to indices
        product_types = metadata["PRODUCT_TYPE_ID"].map(self.product_type_map).values
        targets = metadata["PRODUCT_LENGTH"].values
        product_ids = metadata["PRODUCT_ID"].values
        
        # Load embeddings
        embeddings = self._load_embeddings("train")
        
        # Shuffle and split
        rng = np.random.default_rng(self.config.system.seed)
        indices = rng.permutation(n_samples)
        
        train_end = int(self.config.data.train_ratio * n_samples)
        val_end = int((self.config.data.train_ratio + self.config.data.val_ratio) * n_samples)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"Data splits: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
        
        # Create datasets
        self.train_ds = EmbeddingDataset(
            embeddings={k: v[train_idx] for k, v in embeddings.items()},
            product_types=product_types[train_idx],
            targets=targets[train_idx],
            product_ids=product_ids[train_idx],
        )
        
        self.val_ds = EmbeddingDataset(
            embeddings={k: v[val_idx] for k, v in embeddings.items()},
            product_types=product_types[val_idx],
            targets=targets[val_idx],
            product_ids=product_ids[val_idx],
        )
        
        self.test_ds = EmbeddingDataset(
            embeddings={k: v[test_idx] for k, v in embeddings.items()},
            product_types=product_types[test_idx],
            targets=targets[test_idx],
            product_ids=product_ids[test_idx],
        )
        
        # Store length statistics for post-processing
        self._compute_length_statistics(targets[train_idx], product_types[train_idx])
        
    def _load_embeddings(self, split: str) -> dict[str, np.ndarray]:
        """Load embeddings from active models."""
        embeddings = {}
        
        for model_name in self.active_models:
            filepath = self.embedding_dir / f"{model_name}_{split}.npy"
            
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Embedding file not found: {filepath}\n"
                    f"Run: python scripts/extract_embeddings.py --split {split}"
                )
            
            emb = np.load(filepath, mmap_mode="r")
            embeddings[model_name] = emb
            print(f"  Loaded {model_name}: {emb.shape}")
            
        return embeddings
    
    def _compute_length_statistics(self, targets: np.ndarray, product_types: np.ndarray):
        """Compute length statistics for post-processing."""
        self.all_train_lengths = np.unique(targets)
        self.train_lengths_by_type = {}
        
        for ptype in np.unique(product_types):
            mask = product_types == ptype
            self.train_lengths_by_type[int(ptype)] = np.unique(targets[mask])
            
        print(f"  Unique lengths: {len(self.all_train_lengths):,}")
        print(f"  Product types: {len(self.train_lengths_by_type):,}")
    
    def get_inference_dataset(self) -> EmbeddingDataset:
        """Load test set for final predictions."""
        metadata = pd.read_parquet(self.embedding_dir / "metadata_test.parquet")
        
        product_types = metadata["PRODUCT_TYPE_ID"].map(
            lambda x: self.product_type_map.get(x, 0)
        ).values
        product_ids = metadata["PRODUCT_ID"].values
        
        embeddings = self._load_embeddings("test")
        
        return EmbeddingDataset(
            embeddings=embeddings,
            product_types=product_types,
            product_ids=product_ids,
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.num_workers,
            pin_memory=True,
            persistent_workers=self.config.system.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=True,
            persistent_workers=self.config.system.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=True,
            persistent_workers=self.config.system.num_workers > 0,
        )

"""PyTorch Dataset for pre-computed embeddings with optional KNN features."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset backed by memory-mapped .npy arrays for efficiency.

    Each sample concatenates embeddings from multiple models, adds a product
    type index, and optionally includes KNN-derived statistical features.
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        indices: np.ndarray,
        product_types: np.ndarray,
        targets: Optional[np.ndarray] = None,
        knn_features: Optional[dict[str, np.ndarray]] = None,
        use_log_target: bool = False,
    ):
        self.embeddings = embeddings
        self.indices = indices
        self.product_types = product_types
        self.targets = targets
        self.knn_features = knn_features
        self.use_log_target = use_log_target
        self._model_names = list(embeddings.keys())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]

        # Concatenate embeddings from all active models
        text_emb = np.concatenate(
            [self.embeddings[name][real_idx] for name in self._model_names]
        )

        item: dict[str, torch.Tensor] = {
            "text_embedding": torch.tensor(text_emb, dtype=torch.float32),
            "product_type": torch.tensor(self.product_types[idx], dtype=torch.long),
        }

        if self.targets is not None:
            target = self.targets[idx]
            if self.use_log_target:
                target = np.log1p(target)
            item["target"] = torch.tensor(target, dtype=torch.float32)

        if self.knn_features is not None:
            knn = np.concatenate(
                [self.knn_features[name][real_idx] for name in sorted(self.knn_features)]
            )
            item["knn_features"] = torch.tensor(knn, dtype=torch.float32)

        return item

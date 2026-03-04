"""Post-processing: snap predictions to valid product lengths seen in training data."""

from dataclasses import dataclass, field

import numpy as np


def snap_to_nearest(preds: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Snap each prediction to the nearest valid length via binary search (O(n log m))."""
    if len(valid_lengths) == 0:
        return preds
    valid_sorted = np.sort(valid_lengths)
    indices = np.searchsorted(valid_sorted, preds)
    left = np.maximum(indices - 1, 0)
    right = np.minimum(indices, len(valid_sorted) - 1)
    left_vals, right_vals = valid_sorted[left], valid_sorted[right]
    return np.where(np.abs(preds - left_vals) <= np.abs(preds - right_vals), left_vals, right_vals)


@dataclass
class Snapper:
    """Snaps continuous predictions to discrete valid product lengths.

    Uses type-specific valid lengths when available, falling back to global
    valid lengths for unseen product types.
    """

    all_valid_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    lengths_by_type: dict[int, np.ndarray] = field(default_factory=dict)
    min_length: float = 1.0
    max_length: float = 5000.0

    def process(self, preds: np.ndarray, product_types: np.ndarray) -> np.ndarray:
        """Full post-processing pipeline: clamp positive → snap by type → clip range."""
        result = np.maximum(preds.copy(), 1e-6)

        if self.lengths_by_type:
            for ptype in np.unique(product_types):
                mask = product_types == ptype
                type_lengths = self.lengths_by_type.get(int(ptype), self.all_valid_lengths)
                if len(type_lengths) > 0:
                    result[mask] = snap_to_nearest(result[mask], type_lengths)
        elif len(self.all_valid_lengths) > 0:
            result = snap_to_nearest(result, self.all_valid_lengths)

        return np.clip(result, self.min_length, self.max_length)


def create_snapper(train_targets: np.ndarray, train_product_types: np.ndarray) -> Snapper:
    """Build a Snapper from training data with precomputed valid lengths per type."""
    all_valid = np.sort(np.unique(train_targets))
    lengths_by_type = {
        int(pt): np.sort(np.unique(train_targets[train_product_types == pt]))
        for pt in np.unique(train_product_types)
    }
    print(f"Snapper: {len(all_valid):,} unique lengths, {len(lengths_by_type):,} product types")
    return Snapper(
        all_valid_lengths=all_valid,
        lengths_by_type=lengths_by_type,
        min_length=float(train_targets.min()),
        max_length=float(train_targets.max()),
    )

"""Post-processing: snap predictions to valid product lengths."""

from dataclasses import dataclass, field
import numpy as np

from ..constants import EPSILON


def _snap_to_nearest(preds: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Snap predictions to nearest valid length using binary search (O(n log m))."""
    if len(valid_lengths) == 0:
        return preds
        
    valid_sorted = np.sort(valid_lengths)
    indices = np.searchsorted(valid_sorted, preds)
    
    left_idx = np.maximum(indices - 1, 0)
    right_idx = np.minimum(indices, len(valid_sorted) - 1)
    
    left_vals, right_vals = valid_sorted[left_idx], valid_sorted[right_idx]
    left_dist, right_dist = np.abs(preds - left_vals), np.abs(preds - right_vals)
    
    return np.where(left_dist <= right_dist, left_vals, right_vals)


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + EPSILON))) * 100)


@dataclass
class Snapper:
    """Snaps predictions to valid product lengths seen in training data."""
    
    all_valid_lengths: np.ndarray | None = None
    lengths_by_type: dict[int, np.ndarray] = field(default_factory=dict)
    min_length: float = 1.0
    max_length: float = 5000.0
    
    def snap_global(self, preds: np.ndarray) -> np.ndarray:
        """Snap all predictions to nearest valid length globally."""
        if self.all_valid_lengths is None or len(self.all_valid_lengths) == 0:
            return preds
        return _snap_to_nearest(preds, self.all_valid_lengths)
    
    def snap_by_type(self, preds: np.ndarray, product_types: np.ndarray) -> np.ndarray:
        """Snap predictions using type-specific valid lengths (more accurate)."""
        if not self.lengths_by_type:
            return self.snap_global(preds)
            
        result = preds.copy()
        for ptype in np.unique(product_types):
            mask = product_types == ptype
            type_lengths = self.lengths_by_type.get(int(ptype))
            
            if type_lengths is not None and len(type_lengths) > 0:
                result[mask] = _snap_to_nearest(preds[mask], type_lengths)
            elif self.all_valid_lengths is not None:
                result[mask] = _snap_to_nearest(preds[mask], self.all_valid_lengths)
                
        return result
    
    def clip(self, preds: np.ndarray) -> np.ndarray:
        return np.clip(preds, self.min_length, self.max_length)
    
    def process(
        self,
        preds: np.ndarray,
        product_types: np.ndarray | None = None,
        use_snapping: bool = True,
        snap_by_type: bool = True,
    ) -> np.ndarray:
        """Full post-processing: ensure positive → snap → clip."""
        result = np.maximum(preds.copy(), EPSILON)
        
        if use_snapping:
            if snap_by_type and product_types is not None:
                result = self.snap_by_type(result, product_types)
            else:
                result = self.snap_global(result)
                
        return self.clip(result)
    
    def evaluate(self, preds: np.ndarray, targets: np.ndarray, product_types: np.ndarray | None = None) -> dict[str, float]:
        """Evaluate snapping impact on MAPE."""
        result = {
            "raw_mape": _compute_mape(targets, preds),
            "global_snap_mape": _compute_mape(targets, self.snap_global(preds)),
        }
        if product_types is not None:
            result["type_snap_mape"] = _compute_mape(targets, self.snap_by_type(preds, product_types))
        return result


def create_snapper(train_targets: np.ndarray, train_product_types: np.ndarray) -> Snapper:
    """Create Snapper from training data with precomputed valid lengths."""
    all_valid = np.sort(np.unique(train_targets))
    
    lengths_by_type = {
        int(ptype): np.sort(np.unique(train_targets[train_product_types == ptype]))
        for ptype in np.unique(train_product_types)
    }
    
    from ..utils.logging import get_logger
    get_logger(__name__).info(f"Snapper: {len(all_valid):,} unique lengths, {len(lengths_by_type):,} product types")
    
    return Snapper(
        all_valid_lengths=all_valid,
        lengths_by_type=lengths_by_type,
        min_length=float(train_targets.min()),
        max_length=float(train_targets.max()),
    )


# Backwards compatibility alias
class PostProcessor(Snapper):
    """Alias for Snapper (backwards compatibility)."""
    
    def __init__(self, **kwargs):
        kwargs.pop('calibrator', None)  # Remove deprecated field
        super().__init__(**kwargs)
    
    def fit_calibrator(self, *args, **kwargs):
        """Deprecated: calibration was removed."""
        pass
    
    def calibrate(self, preds: np.ndarray) -> np.ndarray:
        """No-op: returns predictions unchanged."""
        return preds


def create_postprocessor(
    train_targets: np.ndarray,
    train_product_types: np.ndarray,
    val_preds: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    calibration_method: str = "isotonic",
) -> PostProcessor:
    """Create PostProcessor (calibration args ignored for backwards compatibility)."""
    snapper = create_snapper(train_targets, train_product_types)
    return PostProcessor(
        all_valid_lengths=snapper.all_valid_lengths,
        lengths_by_type=snapper.lengths_by_type,
        min_length=snapper.min_length,
        max_length=snapper.max_length,
    )

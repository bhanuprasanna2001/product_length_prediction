"""
Post-Processing Module
======================
Calibration, snapping, and clipping for prediction refinement.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass


@dataclass
class PostProcessor:
    """Post-processor for model predictions."""
    
    all_valid_lengths: np.ndarray | None = None
    lengths_by_type: dict[int, np.ndarray] | None = None
    calibrator: object | None = None
    min_length: float | None = None
    max_length: float | None = None
    
    def fit_calibrator(
        self,
        val_preds: np.ndarray,
        val_targets: np.ndarray,
        method: str = "isotonic",
    ):
        """Fit calibration model on validation data."""
        if method == "linear":
            self.calibrator = LinearRegression()
            self.calibrator.fit(val_preds.reshape(-1, 1), val_targets)
        elif method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=val_targets.max() * 1.1,
                out_of_bounds="clip",
            )
            self.calibrator.fit(val_preds, val_targets)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
        # Report improvement
        raw_mape = _mape(val_targets, val_preds)
        cal_preds = self.calibrate(val_preds)
        cal_mape = _mape(val_targets, cal_preds)
        
        print(f"Calibration ({method}): {raw_mape:.2f}% → {cal_mape:.2f}%")
        
    def calibrate(self, preds: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return preds
        if isinstance(self.calibrator, LinearRegression):
            return self.calibrator.predict(preds.reshape(-1, 1))
        return self.calibrator.predict(preds)
    
    def snap_global(self, preds: np.ndarray) -> np.ndarray:
        """Snap to nearest valid length globally."""
        if self.all_valid_lengths is None:
            return preds
        valid = self.all_valid_lengths
        diffs = np.abs(valid[:, None] - preds[None, :])
        return valid[np.argmin(diffs, axis=0)]
    
    def snap_by_type(self, preds: np.ndarray, product_types: np.ndarray) -> np.ndarray:
        """Snap to nearest valid length per product type."""
        if self.lengths_by_type is None:
            return self.snap_global(preds)
            
        result = np.copy(preds)
        for ptype in np.unique(product_types):
            mask = product_types == ptype
            valid = self.lengths_by_type.get(int(ptype), self.all_valid_lengths)
            if valid is not None and len(valid) > 0:
                type_preds = preds[mask]
                diffs = np.abs(valid[:, None] - type_preds[None, :])
                result[mask] = valid[np.argmin(diffs, axis=0)]
        return result
    
    def clip(self, preds: np.ndarray) -> np.ndarray:
        if self.min_length is not None and self.max_length is not None:
            return np.clip(preds, self.min_length, self.max_length)
        return preds
    
    def process(
        self,
        preds: np.ndarray,
        product_types: np.ndarray | None = None,
        use_calibration: bool = True,
        use_snapping: bool = True,
        snap_by_type: bool = True,
    ) -> np.ndarray:
        """Full post-processing pipeline."""
        result = preds.copy()
        
        if use_calibration and self.calibrator is not None:
            result = self.calibrate(result)
            
        result = np.maximum(result, 1e-6)
        
        if use_snapping:
            if snap_by_type and product_types is not None:
                result = self.snap_by_type(result, product_types)
            else:
                result = self.snap_global(result)
                
        return self.clip(result)


def create_postprocessor(
    train_targets: np.ndarray,
    train_product_types: np.ndarray,
    val_preds: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    calibration_method: str = "isotonic",
) -> PostProcessor:
    """Factory function to create a configured PostProcessor."""
    pp = PostProcessor()
    
    pp.all_valid_lengths = np.unique(train_targets)
    pp.lengths_by_type = {}
    for ptype in np.unique(train_product_types):
        mask = train_product_types == ptype
        pp.lengths_by_type[int(ptype)] = np.unique(train_targets[mask])
        
    pp.min_length = train_targets.min()
    pp.max_length = train_targets.max()
    
    if val_preds is not None and val_targets is not None:
        pp.fit_calibrator(val_preds, val_targets, method=calibration_method)
        
    print(f"PostProcessor: {len(pp.all_valid_lengths):,} lengths, {len(pp.lengths_by_type):,} types")
    
    return pp


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

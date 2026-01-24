"""
Metrics and Evaluation Utilities
================================
"""

import numpy as np
import wandb


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (competition metric)."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error."""
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    product_types: np.ndarray,
    postprocessor,
    stage: str = "test",
) -> dict[str, float]:
    """
    Comprehensive evaluation with and without post-processing.
    
    Returns dict with raw and post-processed metrics.
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION: {stage.upper()}")
    print(f"{'='*50}")
    
    # Raw metrics
    raw_mape = mape(targets, preds)
    raw_rmsle = rmsle(targets, preds)
    print(f"Raw:  MAPE={raw_mape:.2f}%, RMSLE={raw_rmsle:.4f}")
    
    # Post-processed metrics
    pp_preds = postprocessor.process(preds, product_types)
    pp_mape = mape(targets, pp_preds)
    pp_rmsle = rmsle(targets, pp_preds)
    print(f"Post: MAPE={pp_mape:.2f}%, RMSLE={pp_rmsle:.4f}")
    print(f"Improvement: {raw_mape - pp_mape:.2f}%")
    
    # Log comparison table to W&B
    n = min(100, len(preds))
    idx = np.random.choice(len(preds), n, replace=False)
    
    data = [[
        targets[i], preds[i], pp_preds[i],
        abs(targets[i] - preds[i]) / targets[i] * 100,
        abs(targets[i] - pp_preds[i]) / targets[i] * 100,
    ] for i in idx]
    
    table = wandb.Table(
        columns=["true", "raw_pred", "pp_pred", "raw_pct_err", "pp_pct_err"],
        data=data,
    )
    wandb.log({f"{stage}_comparison": table})
    
    return {
        "raw_mape": raw_mape,
        "raw_rmsle": raw_rmsle,
        "pp_mape": pp_mape,
        "pp_rmsle": pp_rmsle,
    }

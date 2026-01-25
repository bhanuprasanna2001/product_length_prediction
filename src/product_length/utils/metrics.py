"""NumPy-based evaluation metrics."""

import numpy as np
import wandb

from ..constants import EPSILON, SCORE_SCALE, DEFAULT_SAMPLE_SIZE


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE as percentage (0-100+)."""
    y_pred_safe = np.maximum(y_pred, EPSILON)
    return float(np.mean(np.abs((y_true - y_pred_safe) / y_true)) * 100)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error."""
    y_pred_safe = np.maximum(y_pred, EPSILON)
    return float(np.sqrt(np.mean((np.log1p(y_pred_safe) - np.log1p(y_true)) ** 2)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition score: max(0, 100 * (1 - MAPE/100))."""
    return max(0.0, SCORE_SCALE * (1.0 - mape(y_true, y_pred) / 100.0))


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    product_types: np.ndarray,
    postprocessor,
    stage: str = "test",
) -> dict[str, float]:
    """Evaluate raw and post-processed predictions, log comparison to W&B."""
    print(f"\n{'='*50}\nEVALUATION: {stage.upper()}\n{'='*50}")
    
    # Raw metrics
    raw_mape, raw_rmsle, raw_score = mape(targets, preds), rmsle(targets, preds), score(targets, preds)
    print(f"Raw:  MAPE={raw_mape:.2f}%, RMSLE={raw_rmsle:.4f}, Score={raw_score:.2f}")
    
    # Post-processed metrics
    pp_preds = postprocessor.process(preds, product_types)
    pp_mape, pp_rmsle, pp_score = mape(targets, pp_preds), rmsle(targets, pp_preds), score(targets, pp_preds)
    print(f"Post: MAPE={pp_mape:.2f}%, RMSLE={pp_rmsle:.4f}, Score={pp_score:.2f}")
    print(f"Improvement: {raw_mape - pp_mape:.2f}%")
    
    # Log comparison to W&B
    n = min(DEFAULT_SAMPLE_SIZE, len(preds))
    idx = np.random.choice(len(preds), n, replace=False)
    data = [[targets[i], preds[i], pp_preds[i], abs(targets[i] - preds[i]) / targets[i] * 100, abs(targets[i] - pp_preds[i]) / targets[i] * 100] for i in idx]
    wandb.log({f"{stage}_comparison": wandb.Table(columns=["true", "raw_pred", "pp_pred", "raw_pct_err", "pp_pct_err"], data=data)})
    
    return {"raw_mape": raw_mape, "raw_rmsle": raw_rmsle, "raw_score": raw_score, "pp_mape": pp_mape, "pp_rmsle": pp_rmsle, "pp_score": pp_score}

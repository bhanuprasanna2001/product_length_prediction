#!/usr/bin/env python
"""
Evaluate Predictions
====================
Compute MAPE and RMSLE for a submission file.

Usage:
    python scripts/evaluate.py submission.csv ground_truth.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.product_length.utils.metrics import mape, rmsle


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("pred_path", type=str, help="Prediction CSV")
    parser.add_argument("true_path", type=str, help="Ground truth CSV")
    args = parser.parse_args()
    
    pred_df = pd.read_csv(args.pred_path)
    true_df = pd.read_csv(args.true_path)
    
    merged = pred_df.merge(true_df, on="PRODUCT_ID")
    y_true = merged["PRODUCT_LENGTH_y"].values
    y_pred = merged["PRODUCT_LENGTH_x"].values
    
    print(f"MAPE:  {mape(y_true, y_pred):.4f}%")
    print(f"RMSLE: {rmsle(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()

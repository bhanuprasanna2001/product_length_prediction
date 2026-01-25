#!/usr/bin/env python
"""Compute MAPE and RMSLE for a submission file."""

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
    
    merged = pd.read_csv(args.pred_path).merge(pd.read_csv(args.true_path), on="PRODUCT_ID")
    y_true, y_pred = merged["PRODUCT_LENGTH_y"].values, merged["PRODUCT_LENGTH_x"].values
    
    print(f"MAPE:  {mape(y_true, y_pred):.4f}%")
    print(f"RMSLE: {rmsle(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()

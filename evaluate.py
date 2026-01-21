import pandas as pd
import numpy as np


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def evaluate(pred_path: str, true_path: str):
    pred_df = pd.read_csv(pred_path)
    true_df = pd.read_csv(true_path)
    
    merged = pred_df.merge(true_df, on='PRODUCT_ID')
    y_true = merged['PRODUCT_LENGTH_y'].values
    y_pred = merged['PRODUCT_LENGTH_x'].values
    
    print(f"MAPE: {mape(y_true, y_pred):.4f}%")
    print(f"RMSLE: {rmsle(y_true, y_pred):.4f}")


if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1], sys.argv[2])

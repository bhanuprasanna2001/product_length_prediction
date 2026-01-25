#!/usr/bin/env python
"""Train ensemble model using YAML config."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.product_length.config import load_config
from src.product_length.training import train


def main():
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    best_checkpoint, best_mape = train(config)
    
    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Best test MAPE: {best_mape:.2f}%")


if __name__ == "__main__":
    main()

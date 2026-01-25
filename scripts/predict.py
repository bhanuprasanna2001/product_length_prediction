#!/usr/bin/env python
"""Generate submission file with post-processing."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.product_length.config import load_config
from src.product_length.inference import predict


def main():
    parser = argparse.ArgumentParser(description="Generate submission predictions")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--postprocessor", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--no-postprocessing", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.config)
    predict(
        config=config,
        checkpoint_path=args.checkpoint,
        postprocessor_path=args.postprocessor,
        output_path=args.output,
        use_postprocessing=not args.no_postprocessing,
    )


if __name__ == "__main__":
    main()

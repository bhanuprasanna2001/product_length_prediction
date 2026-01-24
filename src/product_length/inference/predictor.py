"""
Prediction Pipeline
===================
Inference and submission generation.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from ..config import Config
from ..data import EmbeddingDataModule
from ..models import EnsembleModel
from .postprocessing import PostProcessor


def collect_predictions(
    model: EnsembleModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions from a dataloader."""
    model.eval()
    all_preds, all_targets, all_types = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            pred = model(
                batch["text_embedding"].to(device),
                batch["product_type"].to(device),
            )
            all_preds.append(pred.cpu().numpy())
            if "target" in batch:
                all_targets.append(batch["target"].numpy())
            all_types.append(batch["product_type"].numpy())
            
    preds = np.concatenate(all_preds)
    types = np.concatenate(all_types)
    targets = np.concatenate(all_targets) if all_targets else None
    
    return preds, targets, types


def predict(
    config: Config,
    checkpoint_path: str,
    postprocessor_path: str | None = None,
    output_path: str = "submission.csv",
    use_postprocessing: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions for submission.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model checkpoint
        postprocessor_path: Path to post-processor pickle
        output_path: Path to save submission CSV
        use_postprocessing: Whether to apply post-processing
        
    Returns:
        Submission DataFrame
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    model = EnsembleModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model.to(device)
    
    # Load data
    dm = EmbeddingDataModule(config)
    dm.setup()
    
    # Get test dataset
    test_ds = dm.get_inference_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    
    # Collect predictions
    print("\nGenerating predictions...")
    all_preds, all_types, all_ids = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            pred = model(
                batch["text_embedding"].to(device),
                batch["product_type"].to(device),
            )
            all_preds.append(pred.cpu().numpy())
            all_types.append(batch["product_type"].numpy())
            all_ids.extend(batch["product_id"])
    
    preds = np.concatenate(all_preds)
    product_types = np.concatenate(all_types)
    
    # Post-processing
    if use_postprocessing:
        pp_path = postprocessor_path or config.checkpoints.dir / "postprocessor.pkl"
        if Path(pp_path).exists():
            print(f"\nApplying post-processing...")
            with open(pp_path, "rb") as f:
                postprocessor = pickle.load(f)
            preds = postprocessor.process(preds, product_types)
    
    # Ensure positive
    preds = np.maximum(preds, 0.1)
    
    # Create submission
    submission = pd.DataFrame({
        "PRODUCT_ID": all_ids,
        "PRODUCT_LENGTH": preds,
    })
    
    print(f"\nSubmission: {len(submission):,} samples")
    print(f"  Range: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"  Mean: {preds.mean():.2f}, Median: {np.median(preds):.2f}")
    
    submission.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return submission

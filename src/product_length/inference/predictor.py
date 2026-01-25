"""Inference pipeline and submission generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..constants import EPSILON
from ..data import EmbeddingDataModule
from ..models import EnsembleModel
from ..utils import get_device
from .postprocessing import PostProcessor


@dataclass
class PredictionResult:
    """Prediction outputs container."""
    predictions: np.ndarray
    targets: np.ndarray | None
    product_types: np.ndarray
    product_ids: list[str] | None = None


def _extract_batch_tensors(batch: dict[str, torch.Tensor], device: torch.device):
    """Extract and move batch tensors to device."""
    embeddings = batch["text_embedding"].to(device)
    product_types = batch["product_type"].to(device)
    knn = batch.get("knn_features")
    return embeddings, product_types, knn.to(device) if knn is not None else None


def collect_predictions(
    model: EnsembleModel,
    dataloader: DataLoader,
    device: torch.device,
    show_progress: bool = False,
) -> PredictionResult:
    """Collect predictions from a dataloader."""
    model.eval()
    preds, targets, types = [], [], []
    
    iterator: Iterable = tqdm(dataloader, desc="Collecting predictions") if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            emb, ptypes, knn = _extract_batch_tensors(batch, device)
            pred = model(emb, ptypes, knn_features=knn)
            
            preds.append(pred.cpu().numpy())
            types.append(batch["product_type"].numpy())
            if "target" in batch:
                targets.append(batch["target"].numpy())
    
    return PredictionResult(
        predictions=np.concatenate(preds),
        targets=np.concatenate(targets) if targets else None,
        product_types=np.concatenate(types),
    )


def _load_postprocessor(path: Path) -> PostProcessor | None:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _infer_knn_dim_from_checkpoint(checkpoint_path: str) -> int:
    """Infer knn_dim from checkpoint state_dict for backwards compatibility."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    
    # Check if knn_proj exists and infer dim from first linear layer
    if "knn_proj.0.weight" in state_dict:
        return state_dict["knn_proj.0.weight"].shape[1]
    return 0


def predict(
    config: Config,
    checkpoint_path: str,
    postprocessor_path: str | None = None,
    output_path: str = "submission.csv",
    use_postprocessing: bool = True,
) -> pd.DataFrame:
    """Generate predictions and save submission CSV."""
    device = get_device()
    
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Post-processing: {'enabled' if use_postprocessing else 'disabled'}")
    print("=" * 60)
    
    # Infer knn_dim from checkpoint for backwards compatibility with older checkpoints
    knn_dim = _infer_knn_dim_from_checkpoint(checkpoint_path)
    print(f"Inferred knn_dim from checkpoint: {knn_dim}")
    
    # Load model
    model = EnsembleModel.load_from_checkpoint(
        checkpoint_path, 
        config=config,
        knn_dim=knn_dim,
        use_log_target=config.features.use_log_target,
    )
    model.eval()
    model.to(device)
    
    print(f"Model loaded: knn_dim={model.knn_dim}, use_log_target={model.use_log_target}")
    
    # Setup data for inference
    dm = EmbeddingDataModule(config)
    dm.setup()
    
    test_ds = dm.get_inference_dataset()
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.system.num_workers,
        pin_memory=True,
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    preds_list, types_list, ids_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            emb, ptypes, knn = _extract_batch_tensors(batch, device)
            pred = model(emb, ptypes, knn_features=knn)
            
            preds_list.append(pred.cpu().numpy())
            types_list.append(batch["product_type"].numpy())
            ids_list.extend(batch["product_id"])
    
    preds = np.concatenate(preds_list)
    product_types = np.concatenate(types_list)
    
    print(f"\nRaw predictions: {len(preds):,} samples, range [{preds.min():.2f}, {preds.max():.2f}]")
    
    # Post-processing
    if use_postprocessing:
        pp_path = Path(postprocessor_path) if postprocessor_path else config.checkpoints.dir / "postprocessor.pkl"
        postprocessor = _load_postprocessor(pp_path)
        
        if postprocessor is not None:
            print(f"Applying post-processing from {pp_path}...")
            preds = postprocessor.process(preds, product_types)
            print(f"Post-processed: range [{preds.min():.2f}, {preds.max():.2f}]")
        else:
            print(f"Warning: Postprocessor not found at {pp_path}")
    
    preds = np.maximum(preds, EPSILON)
    
    # Save submission
    submission = pd.DataFrame({"PRODUCT_ID": ids_list, "PRODUCT_LENGTH": preds})
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission: {output_path}")
    
    return submission

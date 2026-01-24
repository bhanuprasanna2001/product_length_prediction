# Product Length Prediction

Predict physical product length from text descriptions using multilingual embedding ensemble.

**Competition Metric:** MAPE (Mean Absolute Percentage Error)

## Project Structure

```
.
├── configs/
│   └── default.yaml              # Configuration file
├── scripts/
│   ├── extract_embeddings.py     # Step 1: Pre-compute embeddings
│   ├── train.py                  # Step 2: Train model
│   ├── predict.py                # Step 3: Generate submission
│   └── evaluate.py               # Evaluate predictions
├── src/product_length/
│   ├── config.py                 # Configuration management
│   ├── data/
│   │   └── embedding_dataset.py  # DataModule for embeddings
│   ├── models/
│   │   └── ensemble.py           # MLP model architecture
│   ├── training/
│   │   ├── callbacks.py          # W&B logging callbacks
│   │   └── trainer.py            # Training pipeline
│   ├── inference/
│   │   ├── postprocessing.py     # Calibration, snapping
│   │   └── predictor.py          # Prediction pipeline
│   └── utils/
│       ├── metrics.py            # MAPE, RMSLE
│       └── embeddings.py         # Embedding extraction
├── data/
│   ├── embeddings/               # Pre-computed embeddings
│   └── total_sentence_data/      # Raw data
├── checkpoints/                  # Model checkpoints
├── notebooks/                    # EDA notebooks
└── wandb/                        # W&B logs
```

## Quick Start

### 1. Extract Embeddings (one-time, ~2-4 hours)

```bash
python scripts/extract_embeddings.py --split train
python scripts/extract_embeddings.py --split test
```

### 2. Train Model

```bash
python scripts/train.py
```

### 3. Generate Predictions

```bash
python scripts/predict.py --checkpoint checkpoints/ensemble-XX-YY.ckpt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           PRE-COMPUTED MULTILINGUAL EMBEDDINGS              │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────┤
│ MiniLM  │  MPNet  │  LaBSE  │   E5    │ BGE-M3  │ ProdType │
│  384d   │  768d   │  768d   │  768d   │  1024d  │  128d    │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬─────┘
     └─────────┴─────────┴────┬────┴─────────┴─────────┘
                              │ concat (3840d)
                        ┌─────▼─────┐
                        │    MLP    │
                        │ 1024→256  │
                        │  256→64   │
                        │   64→1    │
                        └─────┬─────┘
                              ▼
                           LENGTH
```

## Configuration

Edit `configs/default.yaml` to customize:

- **embeddings.active**: Which models to use
- **model.hidden_dims**: MLP architecture  
- **training.loss_fn**: "huber", "mse", or "mape"
- **training.epochs**: Number of epochs
- **postprocessing**: Calibration and snapping options

## Post-Processing Pipeline

1. **Isotonic Calibration** - Corrects systematic bias
2. **Type-Specific Snapping** - Round to valid lengths per product type
3. **Range Clipping** - Ensure within training data bounds

## Tech Stack

- PyTorch Lightning
- Sentence Transformers (multilingual models)
- Weights & Biases (logging)
- scikit-learn (calibration)

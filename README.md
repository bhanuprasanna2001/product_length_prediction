# 📦 Amazon Product Length Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.0+-orange.svg)](https://lightning.ai/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow.svg)](https://wandb.ai/)

Predict the physical length of Amazon products from text metadata (title, bullet points, description) using a multi-embedding ensemble with KNN retrieval features.

**Competition metric:** MAPE (Mean Absolute Percentage Error)

$$\text{Score} = \max\!\bigl(0,\; 100 \times (1 - \text{MAPE})\bigr)$$

---

## Results

| Experiment | Loss | Log-Target | Val MAPE | Notes |
|---|---|---|---|---|
| Huber baseline | Huber | No | 94% | Initial attempt — wrong objective |
| Direct MAPE | MAPE | No | 59% | Aligned loss with metric |
| SMAPE | SMAPE | Yes | ~266% | Symmetric ≠ better here |
| RMSLE | RMSLE | No | ~55% | Decent but sub-optimal |
| **MAPE + log-target** | **MAPE** | **Yes** | **51.78%** | **Best — score ≈ 48.2** |

**Key insight:** Train for what you're measured on. Switching from Huber → MAPE → MAPE + log-target dropped error by **42 percentage points**.

---

## Architecture

```
Input text ──► 4 Sentence Transformers (pre-computed)
                │
                ├─ MiniLM     (384d)
                ├─ MPNet      (768d)
                ├─ DistilUSE  (512d)
                └─ E5-Small   (384d)
                        │
                        ▼ concat (2048d)
               ┌────────┴────────┐
               │  Product Type   │  KNN Stats (20d)
               │  Embedding      │  ─► MLP proj (32d)
               │  (128d)         │
               └────────┬────────┘
                        │ concat (2208d)
                        ▼
                   ┌─────────┐
                   │   MLP   │
                   │ 1024    │
                   │  256    │
                   │   64    │
                   │    1    │
                   └────┬────┘
                        ▼
                     LENGTH
```

**Parameters:** ~4.1M · **Training:** ~20 min/epoch on T4 GPU

### KNN Retrieval Features

For each product, the K=20 nearest neighbors are found in each embedding space using FAISS. Statistical features (mean, std, median, min, max of neighbor lengths) provide a strong retrieval-augmented signal — products described similarly tend to have similar lengths.

### Post-Processing

Predictions are snapped to the nearest valid product length observed per product type in the training set. This consistently improves MAPE by 0.5–1%.

---

## Project Structure

```
├── configs/
│   └── default.yaml              # All hyperparameters in one file
├── notebooks/
│   ├── eda.ipynb                  # Exploratory data analysis
│   ├── product-length-prediction.ipynb   # Full training notebook (Kaggle)
│   └── extract_embeddings_kaggle.ipynb   # Embedding extraction (Kaggle GPU)
├── scripts/
│   ├── train.py                   # Training entry point
│   ├── predict.py                 # Generate submission CSV
│   ├── extract_embeddings.py      # Pre-compute sentence embeddings
│   └── extract_knn_features.py    # Pre-compute KNN retrieval features
├── src/product_length/
│   ├── config.py                  # Single dataclass config + embedding registry
│   ├── dataset.py                 # PyTorch Dataset (memory-mapped)
│   ├── model.py                   # MLPHead + EnsembleModel (LightningModule)
│   ├── losses.py                  # Loss functions + evaluation metrics
│   ├── postprocessing.py          # Snap predictions to valid lengths
│   └── embeddings.py              # Embedding extraction + FAISS KNN utilities
└── pyproject.toml
```

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Pre-compute embeddings (~30 min on GPU)

```bash
python scripts/extract_embeddings.py --split both
```

### 3. Extract KNN features

```bash
python scripts/extract_knn_features.py --embeddings minilm mpnet distiluse e5small
```

### 4. Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 5. Generate predictions

```bash
python scripts/predict.py --checkpoint checkpoints/best.ckpt
```

---

## Configuration

All hyperparameters live in a single YAML file — no nested configs, no framework overhead:

```yaml
# configs/default.yaml (excerpt)
active_embeddings: [minilm, mpnet, distiluse, e5small]
hidden_dims: [1024, 256, 64]
loss_fn: "mape"
use_log_target: true
batch_size: 512
lr: 1.0e-3
epochs: 30
```

Override via a custom YAML:

```bash
python scripts/train.py --config configs/my_experiment.yaml
```

---

## What I Learned

1. **Loss alignment matters most.** Huber loss optimizes the wrong objective for MAPE evaluation. Directly optimizing MAPE gave the biggest single improvement.

2. **Log-target transform handles skew.** Product lengths span 1–5000+ with heavy right skew. Predicting `log(1 + length)` compresses the range and stabilizes training.

3. **Multi-embedding ensemble > single model.** Concatenating 4 diverse embeddings outperforms any single one. Each model captures different semantic aspects.

4. **KNN retrieval is a strong baseline signal.** "Products described similarly have similar lengths" — this simple insight via FAISS nearest-neighbor features adds consistent value.

5. **Post-processing is free performance.** Snapping to valid lengths per product type costs nothing at inference and improves MAPE by ~1%.

6. **Small targets dominate MAPE.** A 10mm prediction error on a 20mm product is 50% error. Understanding this drives architecture and loss design decisions.

---

## Tech Stack

- **PyTorch Lightning** — training loop, mixed precision, checkpointing
- **Sentence Transformers** — multilingual text embeddings
- **FAISS** — fast approximate nearest neighbor search
- **Weights & Biases** — experiment tracking
- **NumPy** — memory-mapped arrays for efficient data loading

---

## License

MIT

---

<p align="center">
  <i>Built with ☕ and PyTorch Lightning</i>
</p>
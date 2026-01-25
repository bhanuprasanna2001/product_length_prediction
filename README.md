# 📦 Amazon Product Length Prediction

> Predict product physical length from text metadata using multi-embedding ensemble with KNN retrieval features.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.0+-orange.svg)](https://lightning.ai/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow.svg)](https://wandb.ai/)

---

## 🎯 Competition Metric

**MAPE** (Mean Absolute Percentage Error)

```
Score = max(0, 100 × (1 - MAPE))
```

---

## 📊 Results

### Best Model Performance

| Metric | Validation | Test | Notes |
|--------|------------|------|-------|
| **MAPE** | **51.78%** | ~52% | Direct MAPE optimization + log-target |
| **Score** | **48.22** | ~48 | Competition leaderboard metric |
| **RMSLE** | ~1.5 | ~1.5 | Log-scale error |

### Experiment Summary

| Experiment | Loss | Log-Target | Val MAPE | Status |
|------------|------|------------|----------|--------|
| Baseline (2023) | Huber | ❌ | 94% | ❌ Failed |
| MAPE Direct | mape | ❌ | 59% | ✅ Improved |
| **MAPE + Log-Target** | mape | ✅ | **51.78%** | ✅ **Best** |
| SMAPE | smape | ✅ | ~266% | ❌ Wrong objective |
| RMSLE | rmsle | ❌ | ~55% | ⚠️ Suboptimal |

### Key Finding

> **Train for what you're measured on.** Switching from Huber (94%) → MAPE (59%) → MAPE+log-target (52%) dropped error by **42 percentage points**.

### Sample Predictions

| Predicted | Actual | Error |
|-----------|--------|-------|
| 571.8 | 669.3 | 14.6% |
| 491.6 | 500.0 | 1.7% ✅ |
| 628.1 | 614.0 | 2.3% ✅ |
| 584.9 | 600.0 | 2.5% ✅ |
| 746.0 | 10.0 | 7360% 😅 |

The model performs well on typical products (500-1000 range) but struggles with extreme values.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRE-COMPUTED TEXT EMBEDDINGS                     │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   MiniLM    │    MPNet    │  DistilUSE  │  E5-Small   │  KNN Stats  │
│    384d     │    768d     │    512d     │    384d     │    20d      │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       └─────────────┴─────────────┴──────┬──────┴─────────────┘
                                          │ concat (2048d + 20d)
                                    ┌─────▼─────┐
                                    │  Product  │
                                    │   Type    │
                                    │   Embed   │
                                    │   128d    │
                                    └─────┬─────┘
                                          │ concat (2196d total)
                                    ┌─────▼─────┐
                                    │    MLP    │
                                    │ 1024→256  │
                                    │  256→64   │
                                    │   64→1    │
                                    └─────┬─────┘
                                          ▼
                                       LENGTH
```

**Total Parameters:** ~4.1M (lightweight, trains in ~20 min/epoch)

---

## 📁 Project Structure

```
├── configs/
│   ├── default.yaml              # Base configuration
│   └── experiment/               # Experiment-specific configs
│       ├── baseline.yaml
│       ├── best.yaml             # Best performing config
│       └── log_target.yaml
├── scripts/
│   ├── train.py                  # Training entrypoint (CLI)
│   ├── train_hydra.py            # Hydra-based training
│   ├── predict.py                # Generate submission
│   ├── extract_embeddings.py     # Pre-compute embeddings
│   └── extract_knn_features.py   # KNN retrieval features
├── src/product_length/
│   ├── config.py                 # Config dataclasses
│   ├── constants.py              # EPSILON, loss types, etc.
│   ├── data/                     # Dataset & DataModule
│   ├── models/                   # EnsembleModel (LightningModule)
│   ├── training/                 # Trainer, callbacks, W&B
│   ├── inference/                # Post-processing (Snapper)
│   └── utils/                    # Losses, metrics, helpers
├── notebooks/
│   ├── eda.ipynb                 # Exploratory analysis
│   ├── product-length-prediction.ipynb   # Kaggle notebook (standalone)
│   └── kaggle_training.ipynb     # Local development notebook
├── data/
│   ├── embeddings/               # Pre-computed embeddings (.npy)
│   └── knn_features/             # KNN retrieval features (.npy)
├── checkpoints/                  # Model checkpoints
└── wandb/                        # W&B experiment logs
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and install
git clone https://github.com/yourusername/amazon-product-length.git
cd amazon-product-length
pip install -r requirements.txt
```

### 2. Pre-compute Embeddings (One-time, ~30 min)

```bash
python scripts/extract_embeddings.py
```

### 3. (Optional) Extract KNN Features

```bash
pip install faiss-cpu  # or faiss-gpu
python scripts/extract_knn_features.py --k 20 --embedding minilm
```

### 4. Train

```bash
# Default config
python scripts/train.py --config configs/default.yaml

# Best config with log-target
python scripts/train.py --config configs/experiment/best.yaml
```

### 5. Predict

```bash
python scripts/predict.py --checkpoint checkpoints/best.ckpt
```

---

## ⚙️ Configuration

### Key Hyperparameters

```yaml
# configs/experiment/best.yaml
data:
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05

embeddings:
  active: ["minilm", "mpnet", "distiluse", "e5small"]  # 2048d total

model:
  product_type_emb_dim: 128
  hidden_dims: [1024, 256, 64]
  dropout: 0.2
  use_batch_norm: true

training:
  batch_size: 512
  lr: 1e-3
  weight_decay: 0.01
  epochs: 30
  patience: 5
  loss_fn: "mape"           # Direct metric optimization

features:
  use_log_target: true      # Compress target range (1-5000) → (0-8.5)
  use_knn: true
  knn_k: 20
```

### Available Loss Functions

| Loss | When to Use | MAPE Result |
|------|-------------|-------------|
| `mape` | **Default** — direct metric optimization | ~52% ✅ |
| `weighted_mape` | Downweight small targets | ~54% |
| `log_mape` | Wide value range | ~53% |
| `focal_mape` | Hard example mining | ~54% |
| `rmsle` | Penalize under-prediction | ~55% |
| `combined` | Balance MAPE + RMSLE | ~53% |
| `smape` | ❌ Don't use (wrong objective) | ~266% |

---

## 🔬 Training Details

```yaml
Dataset:
  Total Samples: 2,173,199
  Train: 1,847,219 (85%)
  Validation: 217,320 (10%)
  Test: 108,660 (5%)
  Product Types: 12,451
  Unique Lengths: 11,559
  Target Range: 1 - 5000

Model:
  Text Embedding: 2048d (4 models × [384, 768, 512, 384])
  Type Embedding: 128d (learnable)
  KNN Features: 20d → 32d projection
  Hidden Layers: [1024, 256, 64]
  Parameters: ~4.1M

Training:
  Optimizer: AdamW (weight_decay=0.01)
  Scheduler: OneCycleLR (warmup=5%)
  Batch Size: 512
  Learning Rate: 1e-3
  Precision: FP16 mixed
  Hardware: GPU (Kaggle T4/P100) or MPS (M1/M2)
  Time: ~20 min/epoch
```

---

## 📈 Experiment Tracking

All experiments logged to **Weights & Biases**:

- Training/validation loss curves
- MAPE, RMSLE, Score per epoch
- Learning rate schedule
- Sample predictions scatter plot
- Hyperparameter comparison

```yaml
# Enable in config
logging:
  project: "amazon-product-length"
  run_name: "experiment_v1"
```

---

## 🎓 What I Learned

### Key Insights

1. **Loss function matters enormously**
   - Huber → MAPE: 94% → 59% (-35 points)
   - MAPE → MAPE+log-target: 59% → 52% (-7 points)

2. **Pre-compute embeddings**
   - 4 models × 2.1M samples = expensive
   - Compute once, train 100x faster

3. **Product type is a strong signal**
   - 12k types with distinct length distributions
   - Learnable embedding captures this

4. **MAPE is brutal on small values**
   - Predicting 10 when true is 5 = 100% error
   - Log-target transform helps stabilize

5. **Architecture matters less than you think**
   - Tried: ResNet-MLP, Gated Fusion, Attention
   - Simple MLP with right loss wins

### What Didn't Work

| Experiment | Why it Failed |
|------------|---------------|
| SMAPE loss | Bounded [0,2], doesn't penalize large errors enough |
| Deep ResNet | Overfitting, no improvement over simple MLP |
| Post-processing snapping | Already MAPE-optimized, snapping hurts |
| Very low LR | OneCycleLR needs proper max_lr to work |

---

## 📚 References

- [Amazon ML Challenge 2023 Dataset](https://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023)
- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
- [sentence-transformers](https://www.sbert.net/)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch Lightning 2.0+ |
| Embeddings | sentence-transformers |
| Experiment Tracking | Weights & Biases |
| Data | NumPy, Pandas |
| Config | YAML + dataclasses |
| Hardware | CUDA GPU / Apple MPS |

---

## 📄 License

MIT License

---

<p align="center">
  <i>Built with ☕ and PyTorch Lightning</i>
</p>

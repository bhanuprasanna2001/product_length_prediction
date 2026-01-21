# Amazon ML Challenge 2023 - Product Length Prediction

**Goal:** Predict product length from text descriptions. Metric: MAPE.

## Problem

Given product information (title, bullet points, description, product type), predict the physical length of the product.

| Column | Description |
|--------|-------------|
| `TITLE` | Product title |
| `BULLET_POINTS` | Feature bullets |
| `DESCRIPTION` | Full description |
| `PRODUCT_TYPE_ID` | Category identifier |
| `PRODUCT_LENGTH` | **Target** (to predict) |

## My Journey

### 2023 Attempt (Failed)
- Used BERT to encode text → predict length directly
- Ignored `PRODUCT_TYPE_ID` completely
- Result: Expensive, slow, poor performance

### 2026 Reattempt (This Repo)
Key insight: **Product type is also a strong signal** (from EDA, RMSLE improves 0.98 → 0.83 with just type median).

Reference Dataset: [Kaggle Amazon ML Challenge](https://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  TOTAL_SENTENCE │     │ PRODUCT_TYPE_ID  │
│  (combined text)│     │   (categorical)  │
└────────┬────────┘     └────────┬─────────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │ MiniLM  │            │ Embedding │
    │ (384-d) │            │  (64-d)   │
    └────┬────┘            └─────┬─────┘
         │                       │
         └───────┬───────────────┘
                 │ concatenate
          ┌──────▼──────┐
          │   448-d     │
          │  MLP Head   │
          │  256 → 64   │
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │   Length    │
          └─────────────┘
```

## Key Decisions

| Decision | Why |
|----------|-----|
| **Concatenation** (not addition) | Preserves information from both modalities in separate dimensions |
| **MiniLM-L6-v2** (not BERT) | similar quality for sentence embeddings |
| **Learnable type embedding** | Captures category-specific length priors |
| **Log-MSE loss** | Approximates RMSLE; more stable than MAPE optimization |
| **Lower LR for encoder** | Fine-tune pretrained weights gently (0.1x head LR) |

## Why Concatenation Works

Both become dense vectors of floats → can be combined:
```
Text:  [t₀, t₁, ..., t₃₈₃]      # 384 dimensions
Type:  [p₀, p₁, ..., p₆₃]       # 64 dimensions
───────────────────────────────
Combined: [t₀...t₃₈₃, p₀...p₆₃] # 448 dimensions (no collision)
```

The MLP learns which dimensions matter. If text contains explicit dimensions ("Width 4.5 feet"), it uses text. Otherwise, it relies on the product type prior.

## Why RMSLE?

Root Mean Squared Logarithmic Error (RMSLE) because it handles large scales/skewed data, penalty on underestimation, and RMSLE is easier to use as a loss function because of differentiability than the non-differentiable MAPE directly.

## Project Structure

```
├── eda.ipynb              # Exploratory data analysis
├── train.py               # Local training script
├── predict.py             # Generate submission
├── evaluate.py            # MAPE/RMSLE evaluation
├── notebooks/
│   ├── eda.ipynb          # Exploratory data analysis
│   └── train_colab.ipynb  # Kaggle/Colab notebook (self-contained)
└── src/
    ├── config.py          # Hyperparameters
    ├── model.py           # LightningModule
    └── data/
        └── dataset.py     # DataModule
```

## Quick Start

**Local:**
```bash
python train.py
python predict.py checkpoints/best-*.ckpt
```

**Colab or Kaggle:**
1. Upload `train_colab.ipynb`
2. Upload data to Drive
3. Run all cells

## EDA Findings

- **2.25M** train samples, **735K** test samples
- **6,295** unique product types
- Target heavily skewed (range: 1 to 9.4M)
- ~32% products have explicit dimensions in text
- Text features alone: near-zero correlation with target
- Product type alone: significant signal

## Tech Stack

- PyTorch Lightning 2.6
- Transformers (HuggingFace)
- Weights & Biases (logging)
- sentence-transformers


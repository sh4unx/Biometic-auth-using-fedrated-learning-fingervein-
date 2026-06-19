# 🔐 Federated Learning — Finger Vein Biometric Authentication

Federated Siamese-network framework for finger vein biometric authentication, comparing six backbone architectures under identical Federated Learning (FedAvg) conditions on the **MMCBNU_6000** dataset.

## Overview

This project benchmarks six Siamese network architectures for finger vein verification, trained with FedAvg and contrastive loss across multiple simulated clients. The core contribution is **DiagonalAdapter** — a lightweight (1,280-parameter) element-wise scaling vector used in place of a standard MLP projection head — paired with an EfficientNet-B0 backbone.

Target venue: IEEE Access / IEEE Transactions on Biometrics, Behavior, and Identity Science.

## Models Compared

| Model | Backbone | Head | Trainable Params |
|---|---|---|---|
| CNN | Custom 5-block CNN (from scratch) | Full model | ~3.5M |
| MobileNetV2 | Pretrained, frozen | 2-layer MLP (512-d) | ~0.5M |
| MobileNetV3-Large | Pretrained, frozen | 2-layer MLP (512-d) | ~0.5M |
| ResNet-34 | Pretrained, frozen | 2-layer MLP (512-d) | ~0.3M |
| DenseNet-161 | Pretrained, frozen | 2-layer MLP (512-d) | ~1.1M |
| **EfficientNet-B0 + DiagonalAdapter** (novelty) | Pretrained, frozen | **DiagonalAdapter** | ~1.3K |

All models are trained under identical Federated Learning conditions:
- Same number of FedAvg rounds
- Same client data splits
- Same contrastive loss, AMP, and gradient clipping
- Three-seed evaluation reported as mean ± std

## Dataset

[MMCBNU_6000](http://multilab.cs.chonbuk.ac.kr/) — a finger vein image dataset used to construct paired (genuine/impostor) samples for Siamese network training. CLAHE preprocessing is applied before feeding images into the network.

> The dataset itself is **not included** in this repository. See [Setup](#setup) below for how to obtain it.

## Notebook Structure

The main notebook (`finger_vein.ipynb`) is organized into the following stages:

1. **Install Dependencies**
2. **Imports & Global Config**
3. **Mount Google Drive** (dataset access)
4. **Preload Images into RAM**
5. **CLAHE Preprocessing & Transforms**
6. **Dataset Diagnostics, Pair Dataset & Client DataLoaders**
7. **Model Definitions** — all 6 architectures
8. **Contrastive Loss**
9. **Training & Evaluation Functions** (generic, shared across all models)
10. **FedAvg Aggregation**
11. **Train All 6 Models** via Federated Learning
12. **Model Comparison** — summary table + visualizations:
    - Multi-metric bar comparison
    - Before vs. After FL comparison
    - Radar chart comparison
    - Per-client FL training curves (combined/best model)
    - ROC / DET curves
13. **Save Models & Export Results**

## Evaluation

Each model is evaluated using:
- Accuracy, EER (Equal Error Rate), and AUC
- ROC and DET curves
- Communication cost (FL rounds × model size)
- Mean ± standard deviation across 3 random seeds

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/sh4unx/Biometic-auth-using-fedrated-learning-fingervein-.git
   cd Biometic-auth-using-fedrated-learning-fingervein-
   ```
2. Open `finger_vein.ipynb` in Google Colab or Jupyter.
3. Mount your Google Drive (or update the data path) and point it to your local copy of the MMCBNU_6000 dataset.
4. Run cells sequentially — Cell 1 installs all dependencies automatically.

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy, matplotlib, scikit-learn
- (Full list installed automatically in Cell 1 of the notebook)

## Citation

If you use this work, please cite the associated paper once published (citation details to be added upon publication).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

# JSAE: Joint Sparse Autoencoder for Cross-Modal Alignment Analysis

This repository contains the code for studying **cross-modal (vision-language) alignment** inside multimodal large language models using **Joint Sparse Autoencoders (JSAE)**. We focus on [LLaVA-v1.6-Mistral-7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) and decompose its intermediate layer activations into interpretable sparse latent features with explicit cross-modal alignment.

## Overview

Understanding how vision and language representations align within multimodal LLMs remains an open question. We propose JSAE — a pair of sparse autoencoders (one for vision tokens, one for text tokens) trained jointly with an alignment loss — to discover shared cross-modal "neurons" (latent features) at a target transformer layer.

The research pipeline consists of four stages:

1. **Data Preparation** — Construct image-caption pairs from COCO Captions with positive, hard-negative, and easy-negative samples.
2. **Linear Probing** — Train linear probes at every transformer layer to measure where vision-language alignment emerges.
3. **JSAE Training** — Train a Joint Sparse Autoencoder on a target layer's activations with a composite loss (reconstruction + L1 sparsity + cosine alignment).
4. **JSAE Analysis** — Analyze the trained JSAE: find highly correlated cross-modal neuron pairs, perform neuron clustering (K-Means + t-SNE), and evaluate cluster quality.

## Project Structure

```
anonymous-jsae/
├── readme.md
├── LICENSE                        # MIT License
└── src/
    ├── get_data.py                # Dataset constructor (positive + hard/easy negatives)
    ├── get_data_only_positive.py  # Dataset constructor (positive samples only)
    ├── probe.py                   # Linear probe trainer for layer-wise alignment analysis
    ├── train_jsae.py              # Joint Sparse Autoencoder training script
    ├── analysis_jsae.py           # JSAE analysis and visualization script
    └── metrics.py                 # Intervention evaluation metrics
```

## Method

### Joint Sparse Autoencoder (JSAE)

The JSAE model consists of two independent Sparse Autoencoders that are trained jointly:

- **Vision SAE**: Encodes mean-pooled vision token activations into a sparse latent space.
- **Text SAE**: Encodes mean-pooled text token activations into a sparse latent space.

Each SAE has the architecture: `Linear(4096 → 16384) → ReLU → Linear(16384 → 4096)`.

The composite training loss is:

$$\mathcal{L} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{sparse}} \cdot \mathcal{L}_{\text{L1}} + \lambda_{\text{align}} \cdot (1 - \cos(z_v, z_t))$$

where the default coefficients are `λ_recon = 1.0`, `λ_sparse = 0.03`, `λ_align = 1.0`.

### Linear Probing

To identify which transformer layer exhibits the strongest cross-modal alignment, we train linear probes at all 31 layers of the LLaVA language model. Each probe takes concatenated features (vision, text, element-wise product, difference, cosine similarity) and predicts whether the image-caption pair is matched. This helps select the optimal target layer for JSAE training (default: layer 13).

## Requirements

- Python >= 3.9
- CUDA-enabled GPU (>= 16GB VRAM recommended for LLaVA in fp16)

### Dependencies

```
torch
transformers
datasets
jsonlines
Pillow
numpy
scipy
scikit-learn
scikit-image
matplotlib
seaborn
torchvision
sentence-transformers
tqdm
```

Install all dependencies:

```bash
pip install torch transformers datasets jsonlines Pillow numpy scipy scikit-learn scikit-image matplotlib seaborn torchvision sentence-transformers tqdm
```

## Usage

### Step 1: Prepare Data

Construct the dataset from COCO Captions. Choose one of the following:

```bash
# Full dataset with positive + negative pairs (for linear probing)
python src/get_data.py
# Output: coco_probe_pairs.jsonl, img/

# Positive-only dataset (for JSAE training)
python src/get_data_only_positive.py
# Output: coco_probe_pairs_positive.jsonl, img_positive/
```

### Step 2: Linear Probing (Optional)

Analyze cross-modal alignment across all transformer layers:

```bash
python src/probe.py
# Input:  coco_probe_pairs.jsonl
# Output: probe_checkpoints/, probe_results.png, layer_progression.png
```

### Step 3: Train JSAE

Train the Joint Sparse Autoencoder on the target layer (default: layer 13):

```bash
python src/train_jsae.py
# Input:  coco_probe_pairs_positive.jsonl
# Output: jsae_lm_13_0.03.pth
```

### Step 4: Analyze JSAE

Run cross-modal neuron correlation analysis and clustering:

```bash
python src/analysis_jsae.py
# Input:  coco_probe_pairs_positive.jsonl, trained JSAE checkpoint
# Output: clustering_vision_13_neurons.png, clustering_text_13_neurons.png
```

## Configuration

Key hyperparameters are hardcoded in the scripts. Adjust as needed:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `device` | All scripts | `cuda:7` / `cuda:6` / `cuda:0` | GPU device index |
| `layer` | `train_jsae.py`, `analysis_jsae.py` | `13` | Target transformer layer |
| `latent_dim` | `train_jsae.py` | `16384` (4x hidden size) | SAE latent dimension |
| `sparse_coef` | `train_jsae.py` | `0.03` | L1 sparsity loss weight |
| `align_coef` | `train_jsae.py` | `1.0` | Alignment loss weight |
| `epochs` | `train_jsae.py` | `20` | Training epochs |
| `learning_rate` | `train_jsae.py` | `1e-5` | Learning rate |
| `batch_size` | `train_jsae.py` | `8` | Training batch size |
| `n_clusters` | `analysis_jsae.py` | `20` | K-Means cluster count |

## Analysis Outputs

The analysis script (`analysis_jsae.py`) produces the following:

- **Cross-modal neuron correlations**: Top-100 most correlated (vision neuron, text neuron) pairs, with co-activation examples showing which image-caption concepts each pair responds to.
- **Neuron clustering**: K-Means clustering of neuron activation profiles, visualized via t-SNE projection.
- **Cluster quality metrics**: Silhouette score, inter/intra cluster distance ratio, and activation entropy for each cluster.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

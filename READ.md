# Transformer Core Engine

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#licensing)
[![Built for Research](https://img.shields.io/badge/research-ready-orange)](#overview)

## Highest-impact Transformer reference for GitHub

Transformer Core is a clean-room, first-principles implementation of decoder-only transformer modeling designed for ambitious research, reproducible experiments, and high-impact engineering.

This repository is not just another transformer clone — it is a compact, transparent engine built to showcase best-practice architecture, stable optimization, and production-grade training dynamics in a form that is easy to inspect, extend, and benchmark.

## Why this repo matters

- **Research-grade clarity:** every layer and operation is implemented explicitly so you can reason about gradient flow, attention dynamics, and normalization behavior.
- **Stable training-first design:** pre-LayerNorm backbone, warmup + cosine decay, gradient clipping, and checkpoint resume make this code reliable for real experiments.
- **Production-ready output:** checkpoint serialization includes optimizer state and can be resumed exactly, enabling long-running runs and reproducible results.

## Highlights

- Decoder-only Transformer architecture with causal self-attention
- Pre-LayerNorm residual blocks for stable convergence
- Modular, inspectable implementation of attention, blocks, and head projection
- Advanced training loop with learning-rate scheduling and eval sampling
- Deterministic execution using fixed random seeds
- Checkpoint save / resume pipeline for seamless iterative development

## Quick Start

```bash
python train.py --max_steps 400 --batch_size 32 --learning_rate 3e-4
```

Optional runtime flags:

- `--resume <path>` — continue training from a saved checkpoint
- `--save_dir <dir>` — store checkpoints in a custom directory

## Technical overview

### Core model

The model implements a standard Transformer decoder stack with:

- learned token embeddings and positional embeddings
- multi-head self-attention using explicit causal masking
- feed-forward layers with GELU activation and dropout inside residual blocks
- final layer normalization and linear language-model head

### Optimization strategy

- **Warmup scheduler + cosine decay** for robust early-stage optimization
- **AdamW** optimizer with weight decay
- **Gradient clipping** to prevent exploding gradients
- **Checkpointing** for reproducible long-run experiments

### Inference & sampling

The training script also includes sample-generation support with temperature scaling and top-k filtering, enabling quick quality checks during evaluation.

## What you can build on this repo

- Transformer architecture research and ablation studies
- custom language modeling experiments on synthetic or real datasets
- production-quality training workflows with checkpoint recovery
- education-focused transformer tutorials with direct code inspection

## Impact-ready features for GitHub

- Clean and concise architecture for fast review and contribution
- Practical training utilities that make the repo immediately useful
- Strong reproducibility story with deterministic checkpoints
- Easy entry point for collaborators through CLI-driven training

## Licensing

This repository is distributed under the MIT License.

## Citation

If used for research or product development, please cite:

Sakurai, Yuta. "Transformer Core: A High Fidelity Autoregressive Modeling Engine." GitHub repository, 2026.

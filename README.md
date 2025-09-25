# Sys-Scan Agent MLOps

A proprietary ML pipeline for training security scan models using synthetic data generation.

## 🚀 Overview

This repository provides an end-to-end solution for:

- **Synthetic Data Generation**: Generate realistic security scan findings for model training
- **ML Training Pipeline**: Fine-tune models for local inference
- **GPU Optimization**: Memory-efficient training with hardware acceleration
- **Production Ready**: Scalable architecture for large-scale dataset generation

## 📦 Quick Start

### 1. Generate Training Data

```bash
# Generate synthetic dataset
./generate_massive_dataset.sh
```

### 2. Train Model

```bash
cd ml_pipeline
pip install -r requirements_keras.txt
python train.py
```

## 📋 Requirements

- **Python**: 3.7+
- **Memory**: 8GB+ recommended
- **Storage**: 10GB+ for datasets
- **GPU**: NVIDIA GPU recommended (CPU fallback available)

## 📝 License

This project is licensed under the Business Source License 1.1. See [LICENSE](LICENSE) for details.

- ✅ **Allowed**: Non-production evaluation, academic research, personal use
- ❌ **Restricted**: SaaS/hosted offerings, production use without commercial agreement
- 📅 **Change Date**: 2028-01-01 (Apache 2.0)

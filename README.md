# Sys-Scan Agent MLOps

A comprehensive ML pipeline for training security scan models using synthetic data generation, featuring Keras 3 with Lion optimizer for efficient GPU training.

## 🚀 Overview

This repository provides an end-to-end solution for:

- **Synthetic Data Generation**: Generate realistic security scan findings for model training
- **ML Training Pipeline**: Fine-tune models using Keras 3 with Lion optimizer
- **GPU Optimization**: Multi-backend support with memory-efficient training
- **Production Ready**: Scalable architecture for large-scale dataset generation

## 🏗️ Architecture

```
├── synthetic_data/          # Synthetic data generation pipeline
│   ├── producers/          # 8 specialized security finding generators
│   ├── correlations/       # Relationship analysis between findings
│   └── verification/       # Quality assurance and validation
├── ml_pipeline/            # Machine learning training components
│   ├── train.py           # Keras 3 training with Lion optimizer
│   ├── preprocess.py      # Data preprocessing pipeline
│   └── requirements/      # Environment configurations
└── massive_datasets/       # Generated training datasets
```

## 📦 Quick Start

### 1. Generate Training Data

```bash
# Generate massive synthetic dataset (~120K findings, ~2 hours)
./generate_massive_dataset.sh

# Custom configuration
./generate_massive_dataset.sh --batch-size 2500 --max-batches 20 --max-hours 1.0
```

### 2. Train Model

```bash
cd ml_pipeline
pip install -r requirements_keras.txt
export KERAS_BACKEND=tensorflow  # or 'jax' or 'torch'
python train.py
```

## 🎯 Key Features

### Synthetic Data Generation

- **8 Producer Types**: Process, network, filesystem, kernel, IOC, MAC, SUID, modules
- **Correlation Analysis**: Process-network, filesystem, and kernel relationships
- **Quality Verification**: Schema validation, coherence checking, realism assessment
- **Parallel Processing**: GPU-optimized with resource monitoring
- **Massive Scale**: Generate 100K+ findings efficiently

### ML Training Pipeline

- **Keras 3 Multi-Backend**: TensorFlow, JAX, or PyTorch support
- **Lion Optimizer**: State-of-the-art optimizer for better convergence
- **GPU Optimization**: Memory growth, mixed precision, early stopping
- **HuggingFace Integration**: Seamless model and tokenizer handling

## 📊 Data Producers

| Producer | Purpose | Example Findings |
|----------|---------|------------------|
| **Process** | System process analysis | Suspicious processes, malware patterns |
| **Network** | Network security scanning | Unusual connections, port scanning |
| **Filesystem** | File permission analysis | World-writable files, SUID binaries |
| **Kernel** | System parameter checking | Security sysctl settings, module analysis |
| **IOC** | Threat indicators | Deleted executables, malicious patterns |
| **MAC** | Access control verification | AppArmor/SELinux status |

## 🔧 Usage Examples

### Generate Custom Dataset

```python
from synthetic_data_pipeline import run_synthetic_data_pipeline

# Generate balanced dataset
result = run_synthetic_data_pipeline(
    output_path="training_data.json",
    producer_counts={
        "processes": 50,
        "network": 30,
        "kernel_params": 20
    },
    compress=True
)
```

### Train with Keras 3

```python
from train import train_with_keras

# Configure training
history = train_with_keras(
    model_name="meta-llama/Llama-2-7b-hf",
    batch_size=4,
    epochs=3,
    learning_rate=2e-4
)
```

### Advanced Pipeline Control

```python
from synthetic_data_pipeline import SyntheticDataPipeline

pipeline = SyntheticDataPipeline(conservative_parallel=True)
result = pipeline.execute_pipeline(
    producer_counts={"processes": 100, "network": 50},
    output_path="dataset.json",
    save_intermediate=True
)
```

## ⚙️ Configuration

### Environment Setup

```bash
# Choose backend (TensorFlow recommended)
export KERAS_BACKEND=tensorflow

# GPU memory growth (optional)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Mixed precision (optional)
export TF_ENABLE_MIXED_PRECISION=1
```

### Dataset Generation Options

- `--batch-size`: Findings per batch (default: 5000)
- `--max-batches`: Total batches to generate (default: 24)
- `--max-hours`: Runtime limit (default: 2.0)
- `--gpu`: Enable GPU optimization (default: enabled)
- `--conservative`: Use conservative parallel processing

## 📋 Requirements

- **Python**: 3.7+
- **Memory**: 8GB+ recommended, 16GB+ for large datasets
- **Storage**: 10GB+ for massive datasets
- **GPU**: NVIDIA GPU recommended (CPU fallback available)

### Dependencies

```bash
# Core requirements
pip install transformers torch accelerate datasets

# Keras 3 (choose one backend)
pip install tensorflow  # or jax or torch
pip install keras

# Optional (for full functionality)
pip install psutil nvidia-ml-py
```

## 📈 Performance

| Configuration | Findings/Hour | Memory | CPU Usage |
|---------------|---------------|--------|-----------|
| GPU Optimized | 30K-50K | 4-8GB | 60-90% |
| CPU Only | 10K-20K | 2-4GB | 80-100% |
| Conservative | 15K-25K | 2-6GB | 40-70% |

## 🎯 Use Cases

- **Model Fine-tuning**: Generate diverse training data for security models
- **Local Inference**: Replace API calls with fine-tuned local models
- **Research**: Study security finding patterns and correlations
- **Testing**: Validate ML pipelines with realistic synthetic data

## 📝 License

This project is licensed under the Business Source License 1.1. See [LICENSE](LICENSE) for details.

- ✅ **Allowed**: Non-production evaluation, academic research, personal use
- ❌ **Restricted**: SaaS/hosted offerings, production use without commercial agreement
- 📅 **Change Date**: 2028-01-01 (Apache 2.0)

## 🤝 Contributing

Contributions welcome! Please see individual component READMEs for detailed documentation:

- [Synthetic Data Pipeline](synthetic_data/README.md)
- [Keras Training](ml_pipeline/README_keras.md)
- [Dataset Generation](DATASET_GENERATION_README.md)

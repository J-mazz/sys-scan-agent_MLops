# Keras 3 Lion Optimizer Training

This module provides fine-tuning capabilities using Keras 3 with the Lion optimizer for GPU-accelerated training.

## Features

- **Keras 3 Multi-Backend**: Supports TensorFlow, JAX, or PyTorch backends
- **Lion Optimizer**: State-of-the-art optimizer for efficient training
- **GPU Optimization**: Memory growth and mixed precision support
- **HuggingFace Integration**: Seamless model and tokenizer loading

## Setup

1. Install dependencies:
```bash
pip install -r requirements_keras.txt
```

2. Set your preferred backend:
```bash
export KERAS_BACKEND=tensorflow  # or 'jax' or 'torch'
```

## Usage

```python
from train import train_with_keras

# Run training
history = train_with_keras()
```

## Lion Optimizer

The Lion optimizer (EvoLved Sign Momentum) provides:
- Better convergence than Adam
- Reduced memory usage
- Improved stability
- Weight decay regularization

## Configuration

- **Backend**: Set via `KERAS_BACKEND` environment variable
- **Batch Size**: 4 (optimized for GPU memory)
- **Sequence Length**: 512 tokens
- **Learning Rate**: 2e-4 with Lion optimizer
- **Weight Decay**: 0.01

## Output

- Model saved in Keras format: `sys-scan-llama-agent-keras3-lion/`
- Checkpoints: `./checkpoints/`
- Training logs: `./logs/` (TensorBoard compatible)
- Tokenizer: Saved alongside model

## Performance Notes

- Uses mixed precision training when available
- Automatic GPU memory growth
- Early stopping and learning rate scheduling
- Comprehensive training metrics

# Sys-Scan Embedded Agent MLOps

ML pipeline for training specialized security analyst models using synthetic data generation and fine-tuning.

## 🚀 Overview

This repository provides an end-to-end solution for:

- **Synthetic Data Generation**: Generate realistic security scan findings for model training
- **Fine-Tuning Pipeline**: Fine-tune Qwen3-4B-Instruct-2507 model using Unsloth
- **Embedded Inference**: Deploy optimized models for local security analysis in air-gapped environments

## Current Strategy

The pipeline uses modern techniques for efficient model training:

- **Model**: Qwen3-VL-4B-Thinking (Vision-Language Model)
- **Fine-Tuning**: SFT (Supervised Fine-Tuning) + GRPO (Reinforcement Learning)
- **Hardware**: NVIDIA GPUs (A100, L4, T4)
- **Libraries**: Unsloth (Dynamic 2.0), vLLM, TRL, PEFT
- **Optimizer**: AdamW (8-bit) for memory-efficient convergence
- **Data**: Synthetic security scan data formatted for instruction tuning with reasoning traces (`<think>` blocks)

## 📝 License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

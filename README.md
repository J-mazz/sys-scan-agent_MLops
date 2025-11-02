# Sys-Scan Embedded Agent MLOps

A proprietary ML pipeline for training specialized security analyst models using synthetic data generation and TPU-accelerated fine-tuning.

## 🚀 Overview

This repository provides an end-to-end solution for:

- **Synthetic Data Generation**: Generate realistic security scan findings for model training
- **TPU Fine-Tuning Pipeline**: Fine-tune Mistral-7B models with LoRA adapters on Google Colab TPUs using nightly Hugging Face libraries
- **Embedded Inference**: Deploy optimized models for local security analysis in air-gapped environments

## Current Strategy

The pipeline uses advanced techniques for efficient model training:

- **Model**: Mistral-7B-Instruct base model
- **Fine-Tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Hardware**: Google Colab TPU v2/v3 with FSDP v2 sharding
- **Libraries**: Nightly builds of Hugging Face Transformers, Datasets, TRL, PEFT, Optimum-TPU
- **Optimizer**: Lion (32-bit) for memory-efficient convergence
- **Data**: Synthetic security scan data formatted for instruction tuning

## 📝 License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

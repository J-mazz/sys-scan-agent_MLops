# ML Fine-Tuning Pipeline for Sys-Scan Security Model

This directory contains a complete fine-tuning pipeline for training a Mistral-7B model on security analysis tasks using synthetic data from sys-scan-graph.

## Overview

The pipeline fine-tunes a 4-bit quantized Mistral-7B model with LoRA adapters to:
- Analyze security findings and assign risk scores
- Identify correlations between security events
- Generate executive summaries and triage recommendations
- Provide actionable remediation steps

All inference runs **locally** with zero external API calls.

## Quick Start

### For Google Colab (TPU)

See `fine_tune_notebook.ipynb` - optimized for TPU v3-8 or v4-8 with massive dataset support.

### For GPU Training

```bash
# Install dependencies
pip install -r requirements_training.txt

# Train
python train.py --data-dir ./synthetic_data --output-dir ./output
```

## Architecture

### Components

1. **Data Preprocessing** (`ml_finetuning_pipeline.py`)
   - Loads synthetic security findings (supports massive datasets)
   - Streaming data loading for memory efficiency
   - Converts to instruction-following format
   - Tokenizes with Mistral chat template

2. **Model Setup** (`ml_finetuning_pipeline.py`)
   - 4-bit quantization (NF4) with bitsandbytes (GPU)
   - BF16 precision (TPU)
   - LoRA adapters on attention and MLP layers
   - Gradient checkpointing for memory efficiency

3. **Optimization** (`lion_optimizer.py`)
   - Lion optimizer implementation
   - Parameter grouping for weight decay

4. **Distillation** (`distillation_trainer.py`)
   - Teacher-student knowledge transfer
   - KL divergence loss
   - Temperature-scaled softmax

5. **Training Orchestration** (TRL `SFTTrainer`)
   - Supervised fine-tuning
   - Completion-only loss (masks instructions)
   - Automatic evaluation
   - TensorBoard logging

## Dataset

The massive_dataset.tar.gz contains:
- 2.5M unique security findings
- LangChain correlation metadata
- Full ground truth schema coverage
- Diverse severity distribution

## Integration with Sys-Scan-Graph

After training:

1. Copy model to sys-scan-graph agent directory
2. Update model path in `sys_scan_agent/llm.py`
3. Test with: `sys-scan-graph analyze --report <report.json>`

## License

Apache License 2.0

#!/usr/bin/env python3
"""
Streamlined Training Script for Sys-Scan Security Model
========================================================

Quick-start fine-tuning script that can be run directly:

    python train.py --data-dir ./synthetic_data --output-dir ./output

Features:
- Automatic data discovery
- GPU memory optimization
- Checkpoint resumption
- TensorBoard logging
- Model evaluation

License: Apache 2.0
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import torch

from ml_finetuning_pipeline import (
    FinetuningConfig,
    SecurityFineTuningPipeline
)
from lion_optimizer import Lion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B for security analysis"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./synthetic_data",
        help="Directory containing synthetic data JSON files"
    )
    parser.add_argument(
        "--data-files",
        type=str,
        nargs="+",
        help="Specific data files to use (optional)"
    )

    # Model arguments
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Teacher model name or path"
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Student model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mistral-7b-security-finetuned",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for models"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    # Optimization arguments
    parser.add_argument(
        "--use-lion",
        action="store_true",
        default=True,
        help="Use Lion optimizer"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--use-distillation",
        action="store_true",
        default=False,
        help="Use teacher-student distillation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--distillation-alpha",
        type=float,
        default=0.5,
        help="Distillation loss weight"
    )

    # Quantization arguments
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 training"
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation"
    )

    return parser.parse_args()


def check_environment():
    """Check compute environment and requirements."""
    logger.info("="*60)
    logger.info("Environment Check")
    logger.info("="*60)

    # Python version
    logger.info(f"Python: {sys.version}")

    # PyTorch
    logger.info(f"PyTorch: {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {total_memory:.2f} GB")

        if total_memory < 20:
            logger.warning(
                "⚠️  GPU has less than 20GB memory. "
                "Consider reducing batch size or sequence length."
            )
    else:
        logger.error("❌ No GPU detected!")
        logger.error("Training will be extremely slow without a GPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    logger.info("="*60)


def discover_data_files(data_dir: str, specific_files=None) -> list:
    """Discover training data files."""
    if specific_files:
        return specific_files

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Find all JSON files except schemas and configs
    data_files = []
    for f in data_path.glob("*.json"):
        name_lower = f.name.lower()
        if "schema" not in name_lower and "config" not in name_lower:
            data_files.append(str(f))

    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(data_files)} data files:")
    for f in data_files:
        logger.info(f"  - {Path(f).name}")

    return data_files


def main():
    """Main training pipeline."""
    args = parse_args()

    # Check environment
    check_environment()

    # Discover data
    data_files = discover_data_files(args.data_dir, args.data_files)

    # Create configuration
    logger.info("Creating pipeline configuration...")

    config = FinetuningConfig(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=not args.no_4bit,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_lion=args.use_lion,
        use_distillation=args.use_distillation,
        distillation_temperature=args.temperature,
        distillation_alpha=args.distillation_alpha,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        bf16=not args.no_bf16,
        seed=args.seed,
    )

    # Save configuration
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Configuration saved to {args.output_dir}/training_config.json")

    # Create pipeline
    logger.info("Initializing fine-tuning pipeline...")
    pipeline = SecurityFineTuningPipeline(config)

    # Run training
    try:
        start_time = datetime.now()
        logger.info("="*60)
        logger.info(f"Starting training at {start_time}")
        logger.info("="*60)

        trainer = pipeline.run_pipeline(data_files)

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("="*60)
        logger.info(f"✓ Training completed in {duration}")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("="*60)

        # Run final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()

        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")

        # Save evaluation results
        with open(f"{args.output_dir}/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"✓ Evaluation results saved to {args.output_dir}/eval_results.json")

        # Training summary
        logger.info("="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        logger.info(f"Start time: {start_time}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Final eval loss: {eval_results.get('eval_loss', 'N/A')}")
        logger.info(f"Model location: {args.output_dir}")
        logger.info("="*60)

        logger.info("\nTo use this model in sys-scan-graph:")
        logger.info(f"1. Copy {args.output_dir} to your sys-scan-graph agent directory")
        logger.info("2. Update model path in sys_scan_agent/llm.py")
        logger.info("3. Test with: sys-scan-graph analyze --report <report.json>\n")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

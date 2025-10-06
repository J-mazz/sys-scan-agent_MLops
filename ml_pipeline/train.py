import os
import torch

# CPU optimizations for C7i
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)

# Disable CUDA-specific optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_from_disk

def train_with_trl():
    """
    Fine-tunes the Mistral-7B-Instruct model using TRL's SFTTrainer.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    dataset_path = "./processed_dataset"
    print(f"Loading pre-processed dataset from {dataset_path}...")
    split_dataset = load_from_disk(dataset_path)

    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    new_model_name = "sys-scan-mistral-agent-trl-sft"

    print(f"Loading tokenizer and model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA configuration for memory efficiency
    lora_config = LoraConfig(
        r=16,                    # Good balance
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",  # Force CPU instead of "auto"
        torch_dtype=torch.float32,  # Use FP32 for CPU training
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Training arguments optimized for CPU training
    training_args = SFTConfig(
        output_dir=f"./checkpoints/{new_model_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,        # Reduce from 8 to 2 for CPU
        gradient_accumulation_steps=4,        # Maintain effective batch size
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim="adamw_torch",                  # Use AdamW instead of Lion for CPU
        fp16=False,                           # Disable for CPU
        bf16=False,                           # Disable for CPU
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,                       # Disable wandb/tensorboard
        dataloader_num_workers=2,             # Reduce from 4 to 2 for CPU
        dataloader_pin_memory=False,          # Disable for CPU
        gradient_checkpointing=True,          # Enable for memory savings
        max_grad_norm=1.0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,                        # Keep False for simplicity
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['validation'],
        tokenizer=tokenizer,
        dataset_text_field="text",  # Assuming dataset has 'text' field
        max_seq_length=512,
        packing=False,
        peft_config=lora_config,  # Add LoRA configuration
    )

    print("\n🚀 Starting fine-tuning with TRL SFTTrainer and AdamW optimizer on CPU...")

    trainer.train()

    print("✅ Fine-tuning completed!")

    print(f"Saving model to {new_model_name}...")
    trainer.save_model(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    print(f"🎉 Model and tokenizer saved to {new_model_name}")

    return trainer

if __name__ == "__main__":
    train_with_trl()

# SageMaker configuration for AWS training
def create_sagemaker_estimator(role_arn=None):
    """
    Create SageMaker PyTorch estimator for CPU training.
    """
    from sagemaker.pytorch import PyTorch

    if role_arn is None:
        # Default role - update with your actual role ARN
        role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role_arn,
        instance_type='ml.m7i.2xlarge',      # Perfect for CPU training!
        instance_count=1,
        framework_version='2.0.1',
        py_version='py310',
        volume_size=150,                     # Larger for your dataset
        max_run=129600,                      # 36 hours max
        use_spot_instances=True,             # Highly recommended for cost savings
        max_wait=14400,                      # 4 hours wait for Spot
        checkpoint_s3_uri='s3://your-bucket/checkpoints/',
        hyperparameters={
            'epochs': 3,
            'learning_rate': 2e-4,
            'batch_size': 4,                 # Can increase with 32GB RAM
            'beta_1': 0.9,
            'beta_2': 0.99,
            'weight_decay': 0.01,
        },
        environment={
            'OMP_NUM_THREADS': '8',
            'MKL_NUM_THREADS': '8',
        }
    )

    return estimator

# Example usage for SageMaker training:
# estimator = create_sagemaker_estimator(role_arn="your-role-arn")
# estimator.fit({
#     'training': 's3://your-bucket/training-data/'
# })
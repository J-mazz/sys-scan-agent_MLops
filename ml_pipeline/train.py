import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.optimization import Lion
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_from_disk

# A100 GPU optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        device_map="auto",  # Use accelerate for device mapping
        torch_dtype=torch.float16,  # Use FP16 for A100 GPU
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Create Lion optimizer (proven efficient for this dataset)
    optimizer = Lion(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )

    # Training arguments optimized for A100 GPU (24GB VRAM)
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{new_model_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,        # Larger batch size for A100
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,        # Effective batch size: 16
        optim="adamw_torch",  # But we'll override with custom Lion optimizer
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,                           # Mixed precision for A100
        bf16=False,                          # Use FP16 instead of BF16
        dataloader_num_workers=4,
        dataloader_pin_memory=True,          # Enable for GPU
        gradient_checkpointing=True,         # Memory optimization
        max_grad_norm=1.0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
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

    # Override optimizer with Lion (proven efficient for this dataset)
    trainer.optimizer = optimizer

    print("\n🚀 Starting fine-tuning with TRL SFTTrainer and Lion optimizer on A100 GPU...")

    trainer.train()

    print("✅ Fine-tuning completed!")

    print(f"Saving model to {new_model_name}...")
    trainer.save_model(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    print(f"🎉 Model and tokenizer saved to {new_model_name}")

    return trainer

if __name__ == "__main__":
    train_with_trl()

# SageMaker configuration for A100 GPU training
def create_sagemaker_estimator(role_arn=None):
    """
    Create SageMaker PyTorch estimator for A100 GPU training.
    """
    from sagemaker.pytorch import PyTorch

    if role_arn is None:
        # Default role - update with your actual role ARN
        role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role_arn,
        instance_type='ml.p4d.24xlarge',      # A100 GPU instance
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
            'batch_size': 8,                 # Larger batch for A100
            'beta_1': 0.9,
            'beta_2': 0.99,
            'weight_decay': 0.01,
        },
        environment={
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        }
    )

    return estimator

# Example usage for SageMaker training:
# estimator = create_sagemaker_estimator(role_arn="your-role-arn")
# estimator.fit({
#     'training': 's3://your-bucket/training-data/'
# })
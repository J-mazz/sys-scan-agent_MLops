import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AdamW
from lion_pytorch import Lion  # Available if needed later
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

# A100 GPU optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Hugging Face authentication (use environment variable)
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

def formatting_func(example):
    """
    Format complete ground truth records for SFT training.
    
    The model learns to generate complete security analysis outputs including:
    - enriched_findings with risk scores and metadata
    - correlations between findings
    - summary narratives (executive, analyst, etc.)
    - recommended actions
    """
    import json
    ground_truth = example
    
    # Create input prompt from enriched findings
    findings_text = json.dumps(ground_truth.get("enriched_findings", []), indent=2)
    
    # Create expected output combining correlations, summaries, and actions
    output_parts = []
    
    if ground_truth.get("correlations"):
        output_parts.append(f"Correlations:\n{json.dumps(ground_truth['correlations'], indent=2)}")
    
    if ground_truth.get("summaries"):
        summaries = {k: v for k, v in ground_truth["summaries"].items() if v is not None}
        if summaries:
            output_parts.append(f"Summaries:\n{json.dumps(summaries, indent=2)}")
    
    if ground_truth.get("actions"):
        output_parts.append(f"Actions:\n{json.dumps(ground_truth['actions'], indent=2)}")
    
    expected_output = "\n\n".join(output_parts) if output_parts else "No additional analysis required."
    
    formatted_text = f"Analyze these security findings and provide a complete assessment:\n\nFindings:\n{findings_text}\n\nAssessment:\n{expected_output}"
    return {"text": formatted_text}

def train_with_trl():
    """
    Fine-tunes the Mistral-7B-Instruct model using TRL's SFTTrainer.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)  # For 2.5M findings
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    dataset_path = "./"
    print(f"Loading pre-processed dataset from {dataset_path}...")
    
    # Check if JSONL files exist, if not run preprocessing
    import os
    if not os.path.exists('train.jsonl') or not os.path.exists('val.jsonl'):
        print("JSONL files not found. Running preprocessing...")
        from preprocess import preprocess_and_save_data
        preprocess_and_save_data()
    
    # Load pre-tokenized JSONL datasets
    train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
    val_dataset = load_dataset('json', data_files='val.jsonl', split='train')
    
    # Remove 'text' column if it exists (from old preprocessing)
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["text"])
    if "text" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns(["text"])
    
    # Combine into DatasetDict format
    from datasets import DatasetDict
    split_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    model_name = "mistralai/Mistral-7B-v0.1"
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

    # Create Lion optimizer (user requested this specific optimizer)
    optimizer = Lion(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )

    # Training arguments optimized for G6 instance (L4 GPU, 24GB VRAM)
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{new_model_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,        # Smaller batch size for G6 instance
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,        # Effective batch size: 16
        optim="adamw_torch",  # We'll override with Lion optimizer
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,                          # Disabled for Lion optimizer compatibility
        bf16=False,                          # Use full precision instead
        dataloader_num_workers=2,            # Fewer workers for G6
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
        max_seq_length=512,
        packing=False,
        peft_config=lora_config,  # Add LoRA configuration
    )

    # Override optimizer with Lion (user requested this optimizer)
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
    try:
        from sagemaker.pytorch import PyTorch
    except ImportError:
        raise ImportError("sagemaker package not available. Install with: pip install sagemaker")

    if role_arn is None:
        # Default role - update with your actual role ARN
        role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role_arn,
        instance_type='ml.g6.2xlarge',      # G6 instance with L4 GPU (cheaper than A100)
        instance_count=1,
        framework_version='2.0.1',
        py_version='py310',
        volume_size=150,                     # Larger for your dataset
        max_run=129600,                      # 36 hours max
        use_spot_instances=True,             # Highly recommended for cost savings
        max_wait=14400,                      # 4 hours wait for Spot
        checkpoint_s3_uri='s3://your-bucket/checkpoints/',
        hyperparameters={
            'epochs': 3,  # Multiple epochs for 2.5M findings
            'learning_rate': 2e-4,
            'batch_size': 4,  # Smaller batch size for G6 instance
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

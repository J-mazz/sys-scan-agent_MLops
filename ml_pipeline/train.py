import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.optimization import Lion
from trl import SFTTrainer
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Use accelerate for device mapping
        torch_dtype="auto"
    )

    # Create custom optimizer (Lion)
    optimizer = Lion(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{new_model_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # Adjust if needed for memory
        optim="adamw_torch",  # But we'll override with custom optimizer
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Mixed precision
        dataloader_num_workers=4,
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
    )

    # Override optimizer
    trainer.optimizer = optimizer

    print("\n🚀 Starting fine-tuning with TRL SFTTrainer and Lion optimizer...")

    trainer.train()

    print("✅ Fine-tuning completed!")

    print(f"Saving model to {new_model_name}...")
    trainer.save_model(new_model_name)
    tokenizer.save_pretrained(new_model_name)

    print(f"🎉 Model and tokenizer saved to {new_model_name}")

    return trainer

if __name__ == "__main__":
    train_with_trl()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_from_disk

def train_on_tpu():
    """
    Loads a pre-processed dataset and runs a fine-tuning job on a TPU.
    """
    dataset_path = "./processed_dataset"
    print(f"Loading pre-processed dataset from {dataset_path}...")
    split_dataset = load_from_disk(dataset_path)
    print("✅ Dataset loaded successfully!")
    print(split_dataset)
    
    model_name = "meta-llama/Meta-Llama-3-8B"
    new_model_name = "sys-scan-llama-agent-tpu"

    lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_arguments = TrainingArguments(
        output_dir="./results_tpu",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        bf16=True, 
        fp16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        dataloader_drop_last=True 
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['validation'],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    print("\n🚀 Starting fine-tuning on TPU...")
    trainer.train()
    print("✅ Fine-tuning completed!")

    trainer.model.save_pretrained(new_model_name)
    print(f"Model saved to {new_model_name}")

if __name__ == "__main__":
    train_on_tpu()
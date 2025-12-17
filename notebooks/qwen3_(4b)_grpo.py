import json
import os
import textwrap

notebook = {
 "cells": [],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "L4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def create_code_cell(source):
    lines = textwrap.dedent(source).strip().split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    }

def create_markdown_cell(source):
    lines = textwrap.dedent(source).strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines]
    }

# --- CELL 1: Intro ---
notebook["cells"].append(create_markdown_cell("""
    # Qwen3 Security Agent: SFT → GRPO Pipeline
    **Full ML Pipeline on L4 GPU**

    This notebook implements a two-stage training pipeline:
    1.  **Stage 1: Supervised Fine-Tuning (SFT)**: Teaches the model the strict JSON schema and formatting required for the security analysis task using `SFTConfig`.
    2.  **Stage 2: GRPO (Reinforcement Learning)**: Optimizes the SFT model's reasoning using `GRPOConfig` to improve Risk Score accuracy and Severity assessment.

    **Configuration:**
    * **Source Install**: Bleeding-edge `unsloth`, `trl`, and `transformers`.
    * **Modern Configs**: Uses `SFTConfig` and `GRPOConfig`.
    * **L4 Optimization**: `bfloat16`, 24GB VRAM.
"""))

# --- CELL 2: Drive Persistence ---
notebook["cells"].append(create_code_cell("""
    import os
    from google.colab import drive

    # Mount Drive
    drive.mount('/content/drive')

    # Paths
    BASE_PATH = "/content/drive/MyDrive/Qwen3_Security_Agent_Pipeline"
    CACHE_DIR = os.path.join(BASE_PATH, "cache")
    SFT_OUTPUT_DIR = os.path.join(BASE_PATH, "sft_checkpoints")
    GRPO_OUTPUT_DIR = os.path.join(BASE_PATH, "grpo_checkpoints")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)
    
    print(f"SFT Checkpoints: {SFT_OUTPUT_DIR}")
    print(f"GRPO Checkpoints: {GRPO_OUTPUT_DIR}")
"""))

# --- CELL 3: Clean Source Install ---
notebook["cells"].append(create_code_cell("""
    %%capture
    import os

    # 1. Clean Environment (Fix Pillow Conflicts)
    !pip uninstall -y pillow
    !pip install "pillow<11.0"

    # 2. Source Installs for Bleeding Edge Features
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install git+https://github.com/huggingface/transformers.git
    !pip install git+https://github.com/huggingface/trl.git
    !pip install git+https://github.com/huggingface/peft.git
    !pip install git+https://github.com/huggingface/accelerate.git

    # 3. Latest Wheels for Compiled Libs
    !pip install --upgrade --no-cache-dir vllm bitsandbytes xformers

    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
"""))

# --- CELL 4: Data Loading ---
notebook["cells"].append(create_code_cell("""
    import json
    from datasets import load_dataset

    # System Prompt (Shared across pipeline)
    SYSTEM_PROMPT = \"\"\"You are an expert security analysis agent.
    1. Analyze the provided system scan finding.
    2. Reason step-by-step about impact, exploitability, and remediation inside <think> tags.
    3. Output the final structured analysis as a valid JSON object inside <answer> tags.

    The JSON output must strictly follow this schema:
    {
      "risk_score": int,      // 0-100
      "severity": "string",   // low, medium, high, critical
      "rationale": "string"   // Brief summary of the risk justification
    }\"\"\"

    print("Streaming dataset...")
    # We load a larger buffer to split between SFT and GRPO
    dataset = load_dataset(
        "jmazz/sys-scan-linux-synthetic",
        data_files="findings/batch_*.jsonl.gz", 
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR
    )
    
    # Materialize 2000 samples
    raw_data = list(dataset.take(2000))
    
    # Split: 500 for SFT (Format Alignment), 1500 for GRPO (Reasoning Optimization)
    sft_data_raw = raw_data[:500]
    grpo_data_raw = raw_data[500:]
    
    print(f"SFT Samples: {len(sft_data_raw)}")
    print(f"GRPO Samples: {len(grpo_data_raw)}")
"""))

# --- CELL 5: Phase 1 - SFT ---
notebook["cells"].append(create_markdown_cell("""
    ## Phase 1: Supervised Fine-Tuning (SFT)
    We first train the model to adhere to the strict JSON output format using standard Supervised Fine-Tuning. 
    This ensures the model knows *how* to speak before we teach it *what* to think.
"""))

notebook["cells"].append(create_code_cell("""
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainingArguments
    import torch

    # 1. Load Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Instruct",
        max_seq_length = 4096,
        load_in_4bit = True,
        max_lora_rank = 64,
        gpu_memory_utilization = 0.8,
        cache_dir = CACHE_DIR,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 2. Format Function for SFT (Standard Chat)
    def formatting_prompts_func(examples):
        convos = []
        texts = []
        for i in range(len(examples["title"])):
            input_payload = {
                "title": examples["title"][i],
                "description": examples["description"][i],
                "metadata": examples["metadata"][i],
                "category": examples["category"][i]
            }
            ground_truth = {
                "risk_score": int(examples["risk_score"][i]),
                "severity": examples["severity"][i],
                "rationale": examples["rationale"][i]
            }
            
            # Unsloth handles the chat template application efficiently
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_payload, indent=2)},
                {"role": "assistant", "content": json.dumps(ground_truth)}
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    # 3. Convert Raw List to Dataset object for SFTTrainer
    from datasets import Dataset
    sft_dataset = Dataset.from_list(sft_data_raw)

    # 4. SFT Configuration (Using Modern SFTConfig)
    sft_config = SFTConfig(
        output_dir = SFT_OUTPUT_DIR,
        dataset_text_field = "text",
        max_seq_length = 4096,
        dataset_num_proc = 2,
        packing = False, # Can be True for speed, but False is safer for complex JSON
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, # Quick format alignment
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = sft_dataset,
        formatting_func = formatting_prompts_func,
        args = sft_config,
        processing_class = tokenizer,
    )

    print("Starting Phase 1: SFT Training...")
    trainer.train()
    
    # 5. Save SFT Adapters
    model.save_lora(os.path.join(SFT_OUTPUT_DIR, "final_sft_adapter"))
    print("SFT Adapter Saved.")
"""))

# --- CELL 6: Phase 2 - GRPO Setup ---
notebook["cells"].append(create_markdown_cell("""
    ## Phase 2: GRPO Training
    We now reload the model with the SFT adapters applied and train it using Group Relative Policy Optimization.
    This phase uses the **Continuous Risk Score** reward to refine the model's judgment.
"""))

notebook["cells"].append(create_code_cell("""
    # 1. Reload Model to Ensure Clean State (Optional but Recommended)
    # For Unsloth, we can just continue, but reloading ensures we aren't carrying over optimizer states weirdly.
    del model, trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load Base Again
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Instruct",
        max_seq_length = 4096,
        load_in_4bit = True,
        max_lora_rank = 64,
        gpu_memory_utilization = 0.8,
        cache_dir = CACHE_DIR,
    )

    # Load the SFT Adapter we just trained
    model.load_lora(os.path.join(SFT_OUTPUT_DIR, "final_sft_adapter"))
    print("Loaded SFT Adapter for GRPO Phase.")

    # 2. Format Data for GRPO (Prompt + Answer separate)
    def format_for_grpo(sample):
        input_payload = {
            "title": sample.get("title"),
            "description": sample.get("description"),
            "metadata": sample.get("metadata", {}),
            "category": sample.get("category", "general")
        }
        ground_truth = {
            "risk_score": int(sample.get("risk_score", 0)),
            "severity": sample.get("severity", "info").lower(),
            "rationale": sample.get("rationale", "")
        }
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_payload, indent=2)}
            ],
            "answer": json.dumps(ground_truth)
        }

    grpo_dataset = Dataset.from_list(grpo_data_raw).map(format_for_grpo)
"""))

# --- CELL 7: GRPO Rewards ---
notebook["cells"].append(create_code_cell("""
    import re
    import json

    # Reward Functions (Same as before)
    def json_format_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            match = re.search(r"<answer>(.*?)</answer>", completion[0]["content"], re.DOTALL)
            if not match:
                rewards.append(-1.0)
                continue
            try:
                json.loads(match.group(1))
                rewards.append(1.0)
            except:
                rewards.append(-0.5)
        return rewards

    def risk_score_accuracy_reward(completions, answer, **kwargs):
        rewards = []
        for completion, gt_str in zip(completions, answer):
            try:
                content = re.search(r"<answer>(.*?)</answer>", completion[0]["content"], re.DOTALL).group(1)
                pred = float(json.loads(content).get("risk_score", -1))
                gt = float(json.loads(gt_str).get("risk_score", -1))
                if not (0 <= pred <= 100):
                    rewards.append(-1.0)
                else:
                    diff = abs(pred - gt)
                    score = 1.0 - (diff / 100.0)
                    if diff <= 5: score += 0.5
                    rewards.append(score)
            except:
                rewards.append(0.0)
        return rewards

    def severity_ordinal_reward(completions, answer, **kwargs):
        ranks = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
        rewards = []
        for completion, gt_str in zip(completions, answer):
            try:
                content = re.search(r"<answer>(.*?)</answer>", completion[0]["content"], re.DOTALL).group(1)
                pred = json.loads(content).get("severity", "").lower()
                gt = json.loads(gt_str).get("severity", "").lower()
                if pred == gt:
                    rewards.append(1.0)
                else:
                    dist = abs(ranks.get(pred,0) - ranks.get(gt,0))
                    rewards.append(max(0.0, 1.0 - (dist * 0.25)))
            except:
                rewards.append(0.0)
        return rewards
"""))

# --- CELL 8: GRPO Training ---
notebook["cells"].append(create_code_cell("""
    from trl import GRPOConfig, GRPOTrainer

    # GRPO Config (Using Modern Class)
    grpo_config = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4, 
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=300, 
        save_steps=50,
        report_to="none",
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        bf16=True, 
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[json_format_reward, risk_score_accuracy_reward, severity_ordinal_reward],
        args=grpo_config,
        train_dataset=grpo_dataset,
    )

    print("Starting Phase 2: GRPO Training...")
    trainer.train()
    
    # Save Final Pipeline Model
    model.save_lora(os.path.join(GRPO_OUTPUT_DIR, "final_pipeline_adapter"))
    print("Pipeline Complete. Model Saved.")
"""))

# Write File
output_file = "Qwen3_Security_Agent_Pipeline.ipynb"
with open(output_file, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {output_file} (SFT -> GRPO Pipeline).")

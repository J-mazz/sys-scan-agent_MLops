import json
import textwrap

# Notebook Structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"version": "3.10.12", "name": "python", "mimetype": "text/x-python", "codemirror_mode": {"name": "ipython", "version": 3}},
        "accelerator": "GPU"
    },
    "nbformat": 4, "nbformat_minor": 5
}

def add_cell(content, type="code"):
    notebook["cells"].append({
        "cell_type": type,
        "metadata": {},
        "source": textwrap.dedent(content).strip().splitlines(keepends=True),
        "outputs": [],
        "execution_count": None
    })

# --- 1. HEADER ---
add_cell("""
    # 🛡️ Qwen3 Security Agent: L4 Speed (vLLM + 4-bit)
    **Hardware:** NVIDIA L4 (24GB) | **Mode:** 4-bit Quantization + vLLM

    This pipeline is tuned for the L4. It creates space for **vLLM acceleration** by using 4-bit quantization,
    solving the "minutes per step" slowness while fitting comfortably in 24GB VRAM.
""", type="markdown")

# --- 2. INSTALL ---
add_cell("""
    # @title 1. Install & Setup (L4 Optimized)
    import os

    # 1. Memory & Speed Tweaks
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # Offload gradients
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 2. Install UV for fast dependency resolution
    !pip install --upgrade -qqq uv

    # 3. Install Unsloth + vLLM
    if "COLAB_" in "".join(os.environ.keys()):
        # L4 uses the newer vLLM kernel (same as A100)
        !uv pip install -qqq --upgrade unsloth vllm==0.10.2 numpy pillow torchvision bitsandbytes xformers triton
    else:
        !pip install unsloth vllm

    # 4. Pin TRL for GRPO stability
    !uv pip install transformers==4.56.2
    !uv pip install --no-deps trl==0.22.2

    print("✅ L4 Environment Ready.")
""")

# --- 3. CONFIG ---
add_cell("""
    # @title 2. Configuration
    from unsloth import FastLanguageModel, PatchFastRL
    import torch
    import os
    import json
    import gc
    import re
    from google.colab import drive, userdata
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset, Dataset

    drive.mount('/content/drive')
    try: os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    except: pass

    class Config:
        base_path = "/content/drive/MyDrive/Qwen3_Security_Agent_Pipeline"
        sft_adapter = os.path.join(base_path, "sft_checkpoints", "final_sft_adapter")
        output_dir = os.path.join(base_path, "grpo_checkpoints")
        cache_dir = os.path.join(base_path, "cache")
        temp_merge_dir = "/content/merged_sft_model_temp"

        # Model Settings (L4 Balanced)
        base_model = "unsloth/Qwen3-4B-Instruct-2507"
        load_in_4bit = True   # <--- 4-bit enables vLLM space on 24GB

        # Training Settings
        batch_size = 8
        grad_accum = 1
        generations = 8       # 8 parallel rollouts (Standard GRPO)
        steps = 300

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
""")

# --- 4. DATA ---
add_cell("""
    # @title 3. Data Pipeline
    SYSTEM_PROMPT = \"\"\"You are an expert security analysis agent.
    1. Analyze the provided system scan finding.
    2. Reason step-by-step about impact, exploitability, and remediation inside <think> tags.
    3. Output the final structured analysis as a valid JSON object inside <answer> tags.

    The JSON output must strictly follow this schema:
    {
      "risk_score": int, "severity": "string", "rationale": "string"
    }\"\"\"

    def prepare_data():
        ds = load_dataset("text", data_files={"train": "hf://datasets/jmazz/sys-scan_synthetic_dataset_v2/train.jsonl"}, streaming=True)

        def safe_parse(ex):
            try:
                d = json.loads(ex['text'])
                d['metadata'] = json.dumps(d.get('metadata', {}))
                return d
            except: return None

        def format_grpo(x):
            meta = json.loads(x['metadata'])
            user = {
                "title": x.get("title"), "description": x.get("description"),
                "metadata": meta, "category": x.get("category", "general")
            }
            gt = {
                "risk_score": int(x.get("risk_score", 0)),
                "severity": x.get("severity", "info").lower(),
                "rationale": x.get("rationale", "")
            }
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user, indent=2)}
                ],
                "answer": json.dumps(gt)
            }

        stream = ds['train'].map(safe_parse, remove_columns=["text"])
        data = [x for x in stream.take(2940) if x is not None]
        return Dataset.from_list(data).map(format_grpo)

    train_dataset = prepare_data()
    print(f"✅ Data Ready: {len(train_dataset)}")
""")

# --- 5. MODEL (4-BIT vLLM) ---
add_cell("""
    # @title 4. Model Setup (4-bit + vLLM)
    def setup_model():
        # 1. Merge SFT (CPU Offload)
        # We perform a high-precision merge first to burn in the SFT weights
        if not os.path.exists(cfg.temp_merge_dir):
            print("🔄 Merging SFT Adapters...")
            base = AutoModelForCausalLM.from_pretrained(cfg.base_model, device_map="cpu", torch_dtype=torch.float16)
            tok = AutoTokenizer.from_pretrained(cfg.base_model)

            model = PeftModel.from_pretrained(base, cfg.sft_adapter)
            model = model.merge_and_unload()

            model.save_pretrained(cfg.temp_merge_dir)
            tok.save_pretrained(cfg.temp_merge_dir)
            del base, model
            gc.collect()

        # 2. Reload with vLLM + 4-bit
        print("🚀 Reloading in 4-bit with vLLM...")

        # Patch for vLLM support in Unsloth
        PatchFastRL("GRPO", FastLanguageModel)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg.temp_merge_dir,
            max_seq_length = 4096,
            load_in_4bit = True,     # <--- 4-bit Quantization
            fast_inference = True,   # <--- vLLM Enabled
            gpu_memory_utilization = 0.6, # 60% of 24GB for Weights + KV Cache
            cache_dir = cfg.cache_dir
        )

        model = FastLanguageModel.get_peft_model(
            model, r=64, lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=3407
        )
        return model, tokenizer

    model, tokenizer = setup_model()
    print("✅ Model Ready (vLLM + 4-bit).")
""")

# --- 6. REWARDS ---
add_cell("""
    # @title 5. Rewards
    def extract_json(t):
        m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
        return json.loads(m.group(1)) if m else None

    def json_format_reward(completions, **kwargs):
        return [1.0 if extract_json(c[0]["content"]) else -1.0 for c in completions]

    def risk_score_reward(completions, answer, **kwargs):
        rews = []
        for c, gt in zip(completions, answer):
            p, g = extract_json(c[0]["content"]), json.loads(gt)
            if not p: rews.append(0.0); continue
            try:
                diff = abs(float(p.get("risk_score", -1)) - float(g.get("risk_score", -1)))
                rews.append(1.0 - (diff/100.0) + (0.5 if diff<=5 else 0))
            except: rews.append(0.0)
        return rews

    def severity_reward(completions, answer, **kwargs):
        rank = {"info":1, "low":2, "medium":3, "high":4, "critical":5}
        rews = []
        for c, gt in zip(completions, answer):
            p, g = extract_json(c[0]["content"]), json.loads(gt)
            if not p: rews.append(0.0); continue
            p_s, g_s = p.get("severity", "").lower(), g.get("severity", "").lower()
            if p_s == g_s: rews.append(1.0)
            else: rews.append(max(0, 1.0 - abs(rank.get(p_s,0)-rank.get(g_s,0))*0.25))
        return rews
""")

# --- 7. TRAIN ---
add_cell("""
    # @title 6. Run Training (vLLM Accelerated)
    print("🚀 Starting GRPO (vLLM Mode)...")

    args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9, adam_beta2=0.99,
        warmup_steps=30, logging_steps=1,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.generations,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=cfg.steps,
        save_steps=50,
        report_to="none",

        # vLLM Settings for L4
        bf16=True, fp16=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.35, # Reserve space for vLLM context
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[json_format_reward, risk_score_reward, severity_reward],
        args=args, train_dataset=train_dataset
    )

    trainer.train()
    model.save_lora(os.path.join(cfg.output_dir, "final_pipeline_adapter"))
    print("✅ Done! Saved to Drive.")
""")

with open("Qwen3_Security_Agent_L4_vLLM.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("✅ L4 vLLM Notebook Generated: Qwen3_Security_Agent_L4_vLLM.ipynb")

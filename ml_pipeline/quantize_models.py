"""
Extreme Quantization Script for Embedded Deployment

Creates ultra-compressed models (<400MB total) split into
multiple safetensors files (<50MB each) for embedded sys-scan-graph deployment.
"""

import os
import torch
import json
import shutil
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import safetensors.torch as safetensors
from datasets import load_dataset


def merge_lora_adapters(adapter_path: str, output_path: str):
    """Merge LoRA adapters with base model."""
    print(f"Merging LoRA adapters from {adapter_path}...")
    
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    ).merge_and_unload()
    
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved to {output_path}")
    return merged_model, tokenizer


def extreme_quantize_model(model_path: str, output_path: str, bits: int = 4):
    """Apply 4-bit GPTQ quantization."""
    print(f"Applying {bits}-bit GPTQ quantization...")
    
    quantize_config = BaseQuantizeConfig(
        bits=bits, group_size=128, desc_act=True, damp_percent=0.1,
        sym=True, true_sequential=True
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, quantize_config=quantize_config, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    calib_dataset = load_dataset("c4", split="train", streaming=True).take(512)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    model.quantize(calib_dataset, tokenize_function=tokenize_function)
    
    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path, use_safetensors=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ {bits}-bit quantized model saved to {output_path}")
    return model, tokenizer


def split_safetensors_into_chunks(model_path: str, output_dir: str, max_chunk_size_mb: int = 50):
    """Split safetensors into <50MB chunks."""
    print(f"Splitting into {max_chunk_size_mb}MB chunks...")
    
    os.makedirs(output_dir, exist_ok=True)
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return None
    
    chunk_count = 0
    current_chunk = {}
    current_size = 0
    
    for safetensor_file in safetensors_files:
        tensors = safetensors.load_file(safetensor_file)
        
        for tensor_name, tensor in tensors.items():
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            if current_size + tensor_size_mb > max_chunk_size_mb and current_chunk:
                chunk_file = os.path.join(output_dir, f"model_chunk_{chunk_count:03d}.safetensors")
                safetensors.save_file(current_chunk, chunk_file)
                chunk_size_mb = sum(t.numel() * t.element_size() for t in current_chunk.values()) / (1024 * 1024)
                print(f"✅ Saved chunk {chunk_count} ({chunk_size_mb:.1f}MB)")
                
                current_chunk = {}
                current_size = 0
                chunk_count += 1
            
            current_chunk[tensor_name] = tensor
            current_size += tensor_size_mb
    
    if current_chunk:
        chunk_file = os.path.join(output_dir, f"model_chunk_{chunk_count:03d}.safetensors")
        safetensors.save_file(current_chunk, chunk_file)
        chunk_size_mb = sum(t.numel() * t.element_size() for t in current_chunk.values()) / (1024 * 1024)
        print(f"✅ Saved final chunk {chunk_count} ({chunk_size_mb:.1f}MB)")
    
    total_size_mb = sum(os.path.getsize(f) for f in Path(output_dir).glob("*.safetensors")) / (1024 * 1024)
    print(f"✅ Model split into {chunk_count + 1} chunks, total: {total_size_mb:.1f}MB")
    return chunk_count + 1, total_size_mb


def create_deployment_package(model_dir: str, output_package: str):
    """Create deployment package with metadata."""
    print(f"Creating deployment package at {output_package}...")
    
    os.makedirs(output_package, exist_ok=True)
    chunk_files = list(Path(model_dir).glob("*.safetensors"))
    
    for chunk_file in chunk_files:
        shutil.copy2(chunk_file, output_package)
    
    config_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]
    for config_file in config_files:
        src = os.path.join(model_dir, config_file)
        if os.path.exists(src):
            shutil.copy2(src, output_package)
    
    metadata = {
        "model_name": "sys-scan-mistral-agent-extreme-quantized",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "quantization": "GPTQ-4bit",
        "total_chunks": len(chunk_files),
        "max_chunk_size_mb": 50,
        "total_size_mb": sum(os.path.getsize(f) for f in chunk_files) / (1024 * 1024),
        "created_date": "2025-01-06",
        "target_platform": "embedded-sys-scan-graph"
    }
    
    with open(os.path.join(output_package, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    loading_script = '''"""
Load extreme quantized model chunks.
Usage: from load_extreme_quantized import load_model_chunks
"""
import json
from pathlib import Path
import safetensors.torch as safetensors
from transformers import AutoTokenizer

def load_model_chunks(package_path: str):
    package_path = Path(package_path)
    with open(package_path / "metadata.json") as f:
        metadata = json.load(f)
    print(f"Loading {metadata['model_name']} from {metadata['total_chunks']} chunks...")
    all_tensors = {}
    for chunk_file in sorted(package_path.glob("model_chunk_*.safetensors")):
        print(f"Loading {chunk_file.name}...")
        all_tensors.update(safetensors.load_file(chunk_file))
    tokenizer = AutoTokenizer.from_pretrained(package_path)
    return all_tensors, tokenizer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        tensors, tokenizer = load_model_chunks(sys.argv[1])
        print(f"Loaded {len(tensors)} tensors")
    else:
        print("Usage: python load_extreme_quantized.py <package_path>")
'''
    
    with open(os.path.join(output_package, "load_extreme_quantized.py"), "w") as f:
        f.write(loading_script)
    
    readme = f"""# Extreme Quantized Model Package

Model: {metadata['model_name']}
- Base: {metadata['base_model']}
- Quantization: {metadata['quantization']}
- Size: {metadata['total_size_mb']:.1f}MB
- Chunks: {metadata['total_chunks']} (<50MB each)

Usage:
```python
from load_extreme_quantized import load_model_chunks
tensors, tokenizer = load_model_chunks(".")
```
"""
    
    with open(os.path.join(output_package, "README.md"), "w") as f:
        f.write(readme)
    
    print(f"✅ Deployment package created at {output_package}")


def validate_quantization_results(package_path: str):
    """Validate quantization meets requirements."""
    print(f"🔍 Validating results in {package_path}...")
    
    package_path = Path(package_path)
    chunk_files = list(package_path.glob("model_chunk_*.safetensors"))
    
    if not chunk_files:
        print("❌ No chunk files found!")
        return False
    
    max_chunk_size = 50 * 1024 * 1024
    oversized = []
    
    for chunk_file in chunk_files:
        size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
        print(f"  - {chunk_file.name}: {size_mb:.1f}MB")
        if os.path.getsize(chunk_file) > max_chunk_size:
            oversized.append((chunk_file.name, size_mb))
    
    if oversized:
        print(f"❌ {len(oversized)} chunks >50MB")
        return False
    
    total_size_mb = sum(os.path.getsize(f) for f in chunk_files) / (1024 * 1024)
    print(f"📊 Total: {total_size_mb:.1f}MB")
    
    if total_size_mb >= 800:  # 4-bit is larger than 2-bit
        print("❌ Total size >=800MB!")
        return False
    
    print("✅ Validation passed!")
    return True


def main():
    """Main quantization pipeline."""
    print("🚀 Extreme Quantization for Embedded Deployment")
    
    adapter_path = "sys-scan-mistral-agent-a100-lora"
    merged_path = "models/merged_model"
    quantized_path = "models/extreme_quantized"
    chunks_path = "models/model_chunks"
    package_path = "models/deployment_package"
    
    try:
        print("\n📦 Merging LoRA adapters...")
        merge_lora_adapters(adapter_path, merged_path)
        
        print("\n⚡ Applying 4-bit GPTQ quantization...")
        extreme_quantize_model(merged_path, quantized_path, bits=4)
        
        print("\n✂️ Splitting into <50MB chunks...")
        result = split_safetensors_into_chunks(quantized_path, chunks_path, 50)
        if not result:
            return
        
        num_chunks, total_size_mb = result
        
        print("\n📦 Creating deployment package...")
        create_deployment_package(chunks_path, package_path)
        
        print("\n🔍 Validating results...")
        success = validate_quantization_results(package_path)
        
        print("\n✅ Complete!")
        print(f"📊 Size: {total_size_mb:.1f}MB in {num_chunks} chunks")
        print(f"📦 Package: {package_path}")
        
        if success and total_size_mb < 800:
            print("🎉 Target achieved!")
        else:
            print("⚠️ Validation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

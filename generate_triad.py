import sys
import os

# Ensure MLOps library is in path
sys.path.append(os.getcwd())
from synthetic_data.synthetic_data_pipeline import run_synthetic_data_pipeline

print("🏭 1. Generating SFT Primer (Broad Coverage)...")
# SFT needs to see EVERY category to learn the tags for all scanner types.
run_synthetic_data_pipeline(
    output_path="raw_sft.json",
    producer_counts={
        "processes": 50, "network": 50, "suid": 50, "kernel_params": 50,
        "modules": 40, "ioc": 40, "mac": 40, "dns": 40,
        "endpoint_behavior": 40, "world_writable": 40
    },
    conservative_parallel=True
)

print("\n🏭 2. Generating GRPO Curriculum (High Complexity)...")
# GRPO needs findings with causal depth. We focus on the heavy hitters.
run_synthetic_data_pipeline(
    output_path="raw_grpo.json",
    producer_counts={
        "processes": 300,  # High complexity
        "network": 300,    # High complexity
        "suid": 200,       # High risk implications
        "kernel_params": 200, 
        "ioc": 200         # Ambiguous correlation chains
    },
    conservative_parallel=True
)

print("\n🏭 3. Generating Evaluation Set (Balanced)...")
# A smaller mirror of the real world.
run_synthetic_data_pipeline(
    output_path="raw_eval.json",
    producer_counts={
        "processes": 30, "network": 30, "suid": 30, "kernel_params": 30,
        "modules": 30, "ioc": 30, "mac": 30, "dns": 30,
        "endpoint_behavior": 30, "world_writable": 30
    },
    conservative_parallel=True
)

print("\n✅ Generation Complete.")
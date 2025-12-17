#!/usr/bin/env python3
"""
Uploads the existing local dataset export to Hugging Face.
Usage: python tools/upload_to_hf.py --token <YOUR_HF_TOKEN>
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

# Configuration
LOCAL_EXPORT_DIR = Path("datasets/hf_export")
REPO_ID = "jmazz/sys-scan-linux-synthetic"  # Target repo

def main():
    parser = argparse.ArgumentParser(description="Upload local dataset to Hugging Face.")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token", required=False)
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: No token provided. Use --token or set HF_TOKEN env var.")
        return

    if not LOCAL_EXPORT_DIR.exists():
        print(f"Error: Local directory {LOCAL_EXPORT_DIR} does not exist.")
        return

    print(f"Uploading {LOCAL_EXPORT_DIR} to {REPO_ID}...")
    
    api = HfApi(token=token)
    
    # Ensure repo exists
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    
    # Upload
    upload_folder(
        folder_path=str(LOCAL_EXPORT_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token
    )
    
    print("✅ Upload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()

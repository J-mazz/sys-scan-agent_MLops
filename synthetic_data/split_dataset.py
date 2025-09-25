#!/usr/bin/env python3
"""
Dataset Splitting Script for Compressed Batches

This script processes compressed batch datasets and splits them into
training, validation, and test sets for ML training.

Usage:
    python split_dataset.py <input_tar.gz> --output-dir <output_directory> --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

The compressed batches contain JSON with gzip-compressed data stored as hex strings.
"""

import argparse
import json
import gzip
import tarfile
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib


def decompress_hex_data(hex_data: str) -> str:
    """Decompress hex-encoded gzip data back to JSON string."""
    try:
        compressed_bytes = bytes.fromhex(hex_data)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        return decompressed_bytes.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to decompress hex data: {e}")


def extract_findings_from_batch(batch_path: Path) -> List[Dict[str, Any]]:
    """Extract findings from a single batch file."""
    try:
        with open(batch_path, 'r') as f:
            batch_data = json.load(f)

        if 'data' not in batch_data:
            print(f"Warning: No 'data' field in {batch_path}")
            return []

        hex_data = batch_data['data']
        decompressed_json = decompress_hex_data(hex_data)
        parsed_data = json.loads(decompressed_json)

        if 'data' not in parsed_data or 'findings' not in parsed_data['data']:
            print(f"Warning: No findings in decompressed data from {batch_path}")
            return []

        findings_by_category = parsed_data['data']['findings']
        all_findings = []

        # Flatten findings from all categories and severity levels
        for category, severity_levels in findings_by_category.items():
            if isinstance(severity_levels, dict):
                for severity, findings_list in severity_levels.items():
                    if isinstance(findings_list, list):
                        # Add category and severity to each finding
                        for finding in findings_list:
                            if isinstance(finding, dict):
                                finding['_category'] = category
                                finding['_severity'] = severity
                                all_findings.append(finding)

        return all_findings

    except Exception as e:
        print(f"Error processing batch {batch_path}: {e}")
        return []


def create_signature(finding: Dict[str, Any]) -> str:
    """Create a signature for deduplication based on finding content."""
    # Use key attributes to create a unique signature
    key_fields = ['rule_id', 'severity', 'description', 'file_path']
    signature_parts = []

    for field in key_fields:
        value = finding.get(field, '')
        signature_parts.append(str(value))

    signature_string = '|'.join(signature_parts)
    return hashlib.md5(signature_string.encode()).hexdigest()


def split_findings(findings: List[Dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float):
    """Split findings into train/val/test sets."""
    # Deduplicate first
    seen_signatures = set()
    unique_findings = []

    for finding in findings:
        signature = create_signature(finding)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_findings.append(finding)

    print(f"Total findings: {len(findings)}, Unique findings: {len(unique_findings)}")

    # Shuffle for random split (deterministic shuffle using signature)
    unique_findings.sort(key=create_signature)

    total = len(unique_findings)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_findings = unique_findings[:train_end]
    val_findings = unique_findings[train_end:val_end]
    test_findings = unique_findings[val_end:]

    return train_findings, val_findings, test_findings


def save_split(output_dir: str, split_name: str, findings: List[Dict[str, Any]]):
    """Save a split to a JSON file."""
    output_path = Path(output_dir) / f"{split_name}.json"

    data = {
        "split": split_name,
        "count": len(findings),
        "findings": findings
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(findings)} findings to {output_path}")


def process_tarball(tarball_path: str, output_dir: str, train_ratio: float, val_ratio: float, test_ratio: float):
    """Process the entire tarball and create training splits."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_findings = []

    print(f"Processing tarball: {tarball_path}")

    with tarfile.open(tarball_path, 'r:gz') as tar:
        # Get all batch files (exclude generation_report.json and other non-batch files)
        batch_files = [member for member in tar.getmembers()
                      if member.isfile() and member.name.endswith('.json')
                      and 'batch_' in member.name and 'generation_report' not in member.name]

        print(f"Found {len(batch_files)} batch files")

        for member in batch_files:
            # Extract to temporary location
            temp_path = Path(output_dir) / "temp_batch.json"
            tar.extract(member, path=output_dir, set_attrs=False)

            extracted_path = Path(output_dir) / member.name

            # Process the extracted file
            findings = extract_findings_from_batch(extracted_path)
            all_findings.extend(findings)

            # Clean up
            extracted_path.unlink(missing_ok=True)

    print(f"Total findings extracted: {len(all_findings)}")

    if not all_findings:
        print("No findings found. Check the batch format.")
        return

    # Split the findings
    train_findings, val_findings, test_findings = split_findings(
        all_findings, train_ratio, val_ratio, test_ratio
    )

    # Save splits
    save_split(output_dir, "train", train_findings)
    save_split(output_dir, "validation", val_findings)
    save_split(output_dir, "test", test_findings)

    print("Dataset splitting complete!")


def main():
    parser = argparse.ArgumentParser(description="Split compressed batch datasets into training sets")
    parser.add_argument("input_tarball", help="Path to the input tar.gz file containing compressed batches")
    parser.add_argument("--output-dir", "-o", default="training_data",
                       help="Output directory for the split datasets (default: training_data)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Ratio of data for training set (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Ratio of data for validation set (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Ratio of data for test set (default: 0.1)")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Split ratios must sum to 1.0, got {total_ratio}")
        return

    if not Path(args.input_tarball).exists():
        print(f"Error: Input tarball {args.input_tarball} does not exist")
        return

    process_tarball(args.input_tarball, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
import json
import os
import gzip
from tqdm import tqdm
from transformers import AutoTokenizer

def formatting_func_for_finding(finding_record):
    """
    Format individual finding records for SFT training.
    
    Each finding gets its own training example with correlation context.
    """
    finding = finding_record["finding"]
    correlations = finding_record["correlations"]
    
    # Format the individual finding
    finding_text = json.dumps(finding, indent=2)
    
    # Include relevant correlations (those mentioning this finding)
    finding_id = finding.get('id', '')
    relevant_correlations = []
    for corr in correlations:
        if finding_id in str(corr.get('source_ids', [])) or finding_id in str(corr.get('target_ids', [])):
            relevant_correlations.append(corr)
    
    correlation_text = ""
    if relevant_correlations:
        correlation_text = f"\n\nRelevant Correlations:\n{json.dumps(relevant_correlations[:3], indent=2)}"  # Limit to 3
    
    # Create analysis prompt
    severity = finding.get('severity', 'unknown')
    category = finding.get('category', 'unknown')
    
    # Generate expected analysis based on finding characteristics
    expected_analysis = generate_finding_analysis(finding, relevant_correlations)
    
    formatted_text = f"Analyze this {severity} severity {category} security finding:\n\n{finding_text}{correlation_text}\n\nAnalysis: {expected_analysis}"
    
    return {"text": formatted_text}

def generate_finding_analysis(finding, correlations):
    """Generate appropriate analysis for a security finding."""
    severity = finding.get('severity', 'unknown')
    risk_score = finding.get('risk_score', 0)
    
    analysis_parts = []
    
    if severity == 'critical':
        analysis_parts.append("CRITICAL: Immediate remediation required")
    elif severity == 'high':
        analysis_parts.append("HIGH: Remediate within 24 hours")
    elif severity == 'medium':
        analysis_parts.append("MEDIUM: Address in regular maintenance cycle")
    else:
        analysis_parts.append("LOW: Monitor and assess impact")
    
    if risk_score > 0.8:
        analysis_parts.append(f"High risk score ({risk_score:.2f}) indicates significant threat")
    elif risk_score > 0.5:
        analysis_parts.append(f"Moderate risk score ({risk_score:.2f}) requires attention")
    
    if correlations:
        analysis_parts.append(f"Correlated with {len(correlations)} other findings - investigate relationships")
    
    return " | ".join(analysis_parts) if analysis_parts else "Review finding details and determine appropriate action"

def flatten_findings(findings_dict):
    """Flatten the nested findings structure into a list for processing."""
    flattened = []
    for scanner_type, severity_groups in findings_dict.items():
        if isinstance(severity_groups, dict):
            for severity, findings_list in severity_groups.items():
                if isinstance(findings_list, list):
                    for finding in findings_list:
                        finding_copy = finding.copy()
                        if 'category' not in finding_copy:
                            finding_copy['category'] = scanner_type
                        flattened.append(finding_copy)
    return flattened

def convert_optimized_to_ground_truth(optimized_data):
    """
    Convert optimized dataset format back to ground truth format for training.
    
    Args:
        optimized_data: Dict with version, findings, correlations, etc.
        
    Returns:
        Ground truth format dict with enriched_findings, correlations, summaries, actions
    """
    try:
        # Preserve the nested structure - DO NOT flatten
        enriched_findings = optimized_data.get('findings', {})
        
        # Extract correlations
        correlations = optimized_data.get('correlations', [])
        
        # Generate basic summaries using flattened findings for processing
        flattened_findings = flatten_findings(enriched_findings)
        summaries = {}  # Placeholder for summaries
        
        # Generate basic actions
        actions = generate_basic_actions(flattened_findings, correlations)
        
        # Generate basic reductions
        reductions = generate_basic_reductions(flattened_findings)
        
        return {
            "version": "ground_truth_v1",
            "enriched_findings": enriched_findings,  # Keep nested structure
            "correlations": correlations,
            "reductions": reductions,
            "summaries": summaries,
            "actions": actions
        }
    except Exception as e:
        print(f"Error converting optimized data: {e}")
        return None

def generate_basic_actions(findings, correlations):
    """Generate basic recommended actions."""
    actions = []
    
    # Group by severity
    critical_findings = [f for f in findings if f.get('severity') == 'critical']
    high_findings = [f for f in findings if f.get('severity') == 'high']
    
    if critical_findings:
        actions.append({
            "id": "critical_remediation",
            "title": "Address Critical Security Findings",
            "description": f"Immediately remediate {len(critical_findings)} critical severity findings",
            "priority": "critical",
            "severity": "critical"
        })
    
    if high_findings:
        actions.append({
            "id": "high_remediation", 
            "title": "Address High Severity Findings",
            "description": f"Remediate {len(high_findings)} high severity findings within 24 hours",
            "priority": "high",
            "severity": "high"
        })
    
    if correlations:
        actions.append({
            "id": "investigate_correlations",
            "title": "Investigate Finding Correlations",
            "description": f"Analyze {len(correlations)} correlations for root cause identification",
            "priority": "medium",
            "severity": "medium"
        })
    
    return actions

def generate_basic_reductions(findings):
    """Generate basic aggregated reductions."""
    # Group findings by category
    category_counts = {}
    for finding in findings:
        category = finding.get('category', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Top findings by risk score
    sorted_findings = sorted(findings, key=lambda x: x.get('risk_score', 0), reverse=True)
    top_findings = sorted_findings[:10]  # Top 10
    
    return {
        "top_findings": [
            {
                "id": f.get('id'),
                "title": f.get('title'),
                "severity": f.get('severity'),
                "risk_score": f.get('risk_score'),
                "probability_actionable": f.get('probability_actionable', 0.5)
            }
            for f in top_findings
        ],
        "category_summary": category_counts
    }

def preprocess_and_save_data():
    """
    Finds, processes, and saves complete ground truth records to JSONL format.
    Extracts full analysis outputs including enriched_findings, correlations, summaries, and actions.
    """
    # --- Define Correct Absolute Paths ---
    # Path to the directory containing your raw data files
    possible_paths = ["../massive_datasets", "./massive_datasets", "/home/sagemaker-user/massive_datasets"]
    dataset_dir = None
    for path in possible_paths:
        if os.path.isdir(path):
            dataset_dir = path
            break
    
    if dataset_dir is None:
        print("❌ ERROR: Could not find massive_datasets directory. Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return
        
    print(f"✅ Found dataset directory at: {dataset_dir}")

    # Path where the final processed dataset will be saved
    output_dir = "."
    train_output_path = os.path.join(output_dir, "train.jsonl")
    val_output_path = os.path.join(output_dir, "validation.jsonl")
        
    print(f"Starting pre-processing from: {dataset_dir}")

    # (The rest of the script remains the same)
    file_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Only process batch files, skip generation reports and other metadata files
            if file.startswith("batch_") and file.endswith(".json") and "generation_report" not in file:
                file_paths.append(os.path.join(root, file))

    all_findings = []
    for file_path in tqdm(file_paths, desc="Processing raw files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = json.load(infile)
                if content.get('compressed'):
                    hex_string = content['data']
                    compressed_bytes = bytes.fromhex(hex_string)
                    decompressed_json_string = gzip.decompress(compressed_bytes).decode('utf-8')
                    decompressed_data = json.loads(decompressed_json_string)
                    
                    # Handle optimized format and extract individual findings
                    if isinstance(decompressed_data, dict) and 'data' in decompressed_data:
                        optimized_data = decompressed_data['data']
                        findings = flatten_findings(optimized_data.get('findings', {}))
                        correlations = optimized_data.get('correlations', [])
                        
                        # Create individual training records for each finding
                        for finding in findings:
                            all_findings.append({
                                "finding": finding,
                                "correlations": correlations,  # Include correlations context
                                "batch_context": optimized_data.get('version', 'unknown')
                            })
                    elif isinstance(decompressed_data, dict) and decompressed_data.get('version') == 'ground_truth_v1':
                        # Handle ground truth format
                        findings = flatten_findings(decompressed_data.get('enriched_findings', {}))
                        correlations = decompressed_data.get('correlations', [])
                        
                        for finding in findings:
                            all_findings.append({
                                "finding": finding,
                                "correlations": correlations,
                                "batch_context": decompressed_data.get('version', 'unknown')
                            })
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")
            continue
    
    print(f"\n✅ Extracted {len(all_findings)} individual findings from all batches.")
    
    if not all_findings:
        print("❌ No findings found to create a dataset. Exiting.")
        return

    # Debug: check first finding
    if all_findings:
        print(f"Sample finding keys: {list(all_findings[0]['finding'].keys())[:5]}")
        print(f"Correlations in context: {len(all_findings[0]['correlations'])}")

    # Split the data (shuffle for individual findings since correlations are preserved per finding)
    import random
    random.shuffle(all_findings)  # Shuffle individual findings
    split_idx = int(0.8 * len(all_findings))
    train_findings = all_findings[:split_idx]
    val_findings = all_findings[split_idx:]
    
    print(f"Split into {len(train_findings)} training and {len(val_findings)} validation findings")
    
    print(f"Split into {len(train_findings)} training and {len(val_findings)} validation findings")
    
    # Load tokenizer for pre-tokenization
    print("Loading tokenizer for pre-tokenization...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Tokenize training data in batches to avoid memory issues
    print("Tokenizing training data...")
    batch_size = 10  # Process in smaller batches
    tokenized_train = []
    for i in tqdm(range(0, len(train_findings), batch_size), desc="Tokenizing train batches"):
        batch_records = train_findings[i:i+batch_size]
        batch_texts = []
        for record in batch_records:
            formatted = formatting_func_for_finding(record)
            batch_texts.append(formatted["text"])
        
        tokenized_batch = tokenizer(batch_texts, truncation=True, max_length=512, padding=False)
        for j, text in enumerate(batch_texts):
            tokenized_train.append({
                "input_ids": tokenized_batch["input_ids"][j],
                "attention_mask": tokenized_batch["attention_mask"][j]
            })
    
    # Tokenize validation data in batches
    print("Tokenizing validation data...")
    tokenized_val = []
    for i in tqdm(range(0, len(val_findings), batch_size), desc="Tokenizing val batches"):
        batch_records = val_findings[i:i+batch_size]
        batch_texts = []
        for record in batch_records:
            formatted = formatting_func_for_finding(record)
            batch_texts.append(formatted["text"])
        
        tokenized_batch = tokenizer(batch_texts, truncation=True, max_length=512, padding=False)
        for j, text in enumerate(batch_texts):
            tokenized_val.append({
                "input_ids": tokenized_batch["input_ids"][j],
                "attention_mask": tokenized_batch["attention_mask"][j]
            })
    
    # Save tokenized data to JSONL format
    print("Saving tokenized training data to JSONL...")
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(tokenized_train, desc="Writing train"):
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print("Saving tokenized validation data to JSONL...")
    with open(val_output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(tokenized_val, desc="Writing validation"):
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n✅ SUCCESS! Data saved to:")
    print(f"  Training: {train_output_path} ({len(train_findings)} records)")
    print(f"  Validation: {val_output_path} ({len(val_findings)} records)")

if __name__ == "__main__":
    preprocess_and_save_data()
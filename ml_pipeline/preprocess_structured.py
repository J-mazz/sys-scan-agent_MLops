"""
Enhanced preprocessing pipeline with messages format for state-of-the-art fine-tuning.

This module restructures synthetic security data from flat JSON dumps to structured
messages format (system/user/assistant), enabling:
- Multi-task learning (risk scoring, correlation detection, action generation)
- Explicit risk subscore training
- Correlation reasoning capabilities
- Leveraging full ground truth schema metadata

Implements Phase 1 of SoTA optimization strategy.
"""

import json
import os
import gzip
import random
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from pathlib import Path


class StructuredPreprocessor:
    """
    Preprocessor that converts raw security findings to structured messages format.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Task distribution for multi-task learning
        self.task_types = ["risk_scoring", "correlation_detection", "action_generation", "comprehensive_analysis"]
        self.task_weights = [0.3, 0.25, 0.25, 0.2]  # Weighted sampling
        
        self.system_prompt = (
            "You are an expert security analyst specializing in Linux system security. "
            "You analyze security scan findings, assess risks with detailed subscores, "
            "identify correlations between findings, and provide actionable remediation guidance. "
            "Your analyses are precise, data-driven, and aligned with industry best practices."
        )
    
    def create_messages_format(self, finding: Dict[str, Any], task_type: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Convert a finding to messages format based on task type.
        
        Args:
            finding: Raw security finding with metadata
            task_type: One of [risk_scoring, correlation_detection, action_generation, comprehensive_analysis]
        
        Returns:
            Dictionary with 'messages' key containing system/user/assistant messages
        """
        
        if task_type == "risk_scoring":
            return self._create_risk_scoring_messages(finding)
        elif task_type == "correlation_detection":
            return self._create_correlation_messages(finding)
        elif task_type == "action_generation":
            return self._create_action_messages(finding)
        elif task_type == "comprehensive_analysis":
            return self._create_comprehensive_messages(finding)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _create_risk_scoring_messages(self, finding: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Create messages for risk scoring task."""
        
        # Extract metadata
        title = finding.get("title", "Unknown Finding")
        severity = finding.get("severity", "unknown")
        description = finding.get("description", "No description available")
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        baseline_status = finding.get("baseline_status", "unknown")
        tags = finding.get("tags", [])
        
        # Extract ground truth
        risk_subscores = finding.get("risk_subscores", {
            "impact": 0.5,
            "exposure": 0.5,
            "anomaly": 0.5,
            "confidence": 0.5
        })
        probability_actionable = finding.get("probability_actionable", 0.5)
        risk_score = finding.get("risk_score", 50)
        
        user_prompt = f"""Analyze this security finding and provide detailed risk assessment:

**Finding Details:**
- Title: {title}
- Severity: {severity}
- Category: {category}
- Description: {description}
- Baseline Status: {baseline_status}
- Tags: {', '.join(tags) if tags else 'None'}

**Task:** Provide risk subscores on a 0.0-1.0 scale:
1. **Impact**: Potential damage if exploited
2. **Exposure**: Attack surface and accessibility
3. **Anomaly**: Deviation from expected baseline
4. **Confidence**: Reliability of the assessment

Also estimate the probability this finding requires immediate action (0.0-1.0)."""

        # Generate reasoning based on severity and metadata
        reasoning = self._generate_risk_reasoning(finding, risk_subscores)
        
        assistant_response = f"""**Risk Assessment:**

**Risk Subscores:**
- Impact: {risk_subscores.get('impact', 0):.2f}
- Exposure: {risk_subscores.get('exposure', 0):.2f}
- Anomaly: {risk_subscores.get('anomaly', 0):.2f}
- Confidence: {risk_subscores.get('confidence', 0):.2f}

**Probability Actionable:** {probability_actionable:.2f}

**Overall Risk Score:** {risk_score}/100

**Reasoning:**
{reasoning}"""

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def _create_correlation_messages(self, finding: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Create messages for correlation detection task."""
        
        title = finding.get("title", "Unknown Finding")
        severity = finding.get("severity", "unknown")
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        risk_score = finding.get("risk_score", 50)
        correlation_refs = finding.get("correlation_refs", [])
        
        finding_summary = {
            'id': finding.get('id', 'unknown'),
            'title': title,
            'severity': severity,
            'category': category,
            'risk_score': risk_score
        }
        
        user_prompt = f"""Analyze this finding for potential correlations with other security issues:

{json.dumps(finding_summary, indent=2)}

**Task:** 
1. Identify what types of related findings should be investigated
2. Describe correlation patterns that would increase risk
3. Recommend investigation priorities"""

        if correlation_refs:
            correlation_analysis = self._generate_correlation_analysis(finding, correlation_refs)
            assistant_response = f"""**Correlation Analysis:**

**Status:** {len(correlation_refs)} related finding(s) detected

**Correlation IDs:** {', '.join(correlation_refs[:5])}
{f"... and {len(correlation_refs) - 5} more" if len(correlation_refs) > 5 else ""}

**Analysis:**
{correlation_analysis}

**Recommended Investigation:**
{self._generate_investigation_steps(finding)}"""
        else:
            assistant_response = f"""**Correlation Analysis:**

**Status:** No direct correlations detected

**Interpretation:** This appears to be an isolated finding. However, consider investigating:

{self._generate_investigation_steps(finding)}

**Note:** Absence of detected correlations may indicate either genuine isolation or insufficient data context."""

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def _create_action_messages(self, finding: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Create messages for action generation task."""
        
        title = finding.get("title", "Unknown Finding")
        severity = finding.get("severity", "unknown")
        description = finding.get("description", "No description available")
        risk_score = finding.get("risk_score", 50)
        probability_actionable = finding.get("probability_actionable", 0.5)
        
        user_prompt = f"""Provide detailed remediation guidance for this security finding:

**Finding:** {title}
**Severity:** {severity} (Risk Score: {risk_score}/100)
**Actionable Probability:** {probability_actionable:.2f}

**Description:** {description}

**Task:** Provide step-by-step remediation actions, prioritized by impact."""

        # Generate actions based on finding metadata
        actions = self._extract_or_generate_actions(finding)
        
        assistant_response = f"""**Remediation Plan:**

**Priority Level:** {"HIGH - Immediate action recommended" if probability_actionable > 0.7 else "MEDIUM - Schedule remediation" if probability_actionable > 0.4 else "LOW - Monitor and assess"}

**Action Steps:**
{actions}

**Validation:**
After implementing remediation, verify:
1. The specific security control is properly configured
2. No unintended side effects on system functionality
3. Continuous monitoring confirms issue resolution

**Timeline:** {"Within 24 hours" if probability_actionable > 0.7 else "Within 1 week" if probability_actionable > 0.4 else "As part of routine maintenance"}"""

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def _create_comprehensive_messages(self, finding: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        """Create messages for comprehensive analysis task."""
        
        # Provide full finding context (but structured, not raw JSON dump)
        finding_context = {
            'title': finding.get('title', 'Unknown'),
            'severity': finding.get('severity', 'unknown'),
            'description': finding.get('description', ''),
            'risk_score': finding.get('risk_score', 50),
            'category': finding.get('metadata', {}).get('scanner_type', 'unknown'),
            'baseline_status': finding.get('baseline_status', 'unknown'),
            'has_correlations': len(finding.get('correlation_refs', [])) > 0
        }
        
        user_prompt = f"""Perform comprehensive security analysis of this finding:

{json.dumps(finding_context, indent=2)}

**Task:** Provide:
1. Risk assessment with detailed subscores
2. Correlation considerations
3. Actionability evaluation
4. Remediation recommendations"""

        # Generate comprehensive analysis
        comprehensive_analysis = self._generate_comprehensive_analysis(finding)
        
        assistant_response = comprehensive_analysis

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    # Helper methods for generating responses
    
    def _generate_risk_reasoning(self, finding: Dict[str, Any], risk_subscores: Dict[str, float]) -> str:
        """Generate reasoning for risk scores."""
        severity = finding.get("severity", "unknown")
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        baseline = finding.get("baseline_status", "unknown")
        
        impact_reasoning = "High impact" if risk_subscores.get("impact", 0) > 0.7 else "Moderate impact" if risk_subscores.get("impact", 0) > 0.4 else "Low impact"
        exposure_reasoning = "Highly exposed" if risk_subscores.get("exposure", 0) > 0.7 else "Moderately exposed" if risk_subscores.get("exposure", 0) > 0.4 else "Low exposure"
        
        reasoning = f"""This {severity} severity finding in the {category} category shows {impact_reasoning.lower()} potential 
and is {exposure_reasoning.lower()} to exploitation. """
        
        if baseline == "deviation":
            reasoning += "The deviation from baseline significantly increases the anomaly score, indicating unexpected system behavior. "
        elif baseline == "new":
            reasoning += "As a newly detected finding, it requires immediate assessment to determine if it represents a genuine security concern. "
        
        reasoning += f"Assessment confidence is {'high' if risk_subscores.get('confidence', 0) > 0.7 else 'moderate' if risk_subscores.get('confidence', 0) > 0.4 else 'low'} based on available data and context."
        
        return reasoning
    
    def _generate_correlation_analysis(self, finding: Dict[str, Any], correlation_refs: List[str]) -> str:
        """Generate correlation analysis narrative."""
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        severity = finding.get("severity", "unknown")
        
        analysis = f"This {category} finding is correlated with {len(correlation_refs)} other security finding(s). "
        
        if severity in ["high", "critical"]:
            analysis += "The high severity combined with multiple correlations suggests a potential attack pattern or systemic vulnerability. "
        
        analysis += "Correlation patterns indicate related security issues that should be investigated together for comprehensive remediation. "
        
        if len(correlation_refs) >= 3:
            analysis += "The high number of correlations may indicate a cascading security issue or coordinated attack vectors."
        
        return analysis
    
    def _generate_investigation_steps(self, finding: Dict[str, Any]) -> str:
        """Generate investigation recommendations."""
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        
        steps = []
        
        if category == "processes":
            steps.append("- Review process lineage and parent-child relationships")
            steps.append("- Check for associated network connections")
            steps.append("- Examine file system access patterns")
        elif category == "network":
            steps.append("- Analyze connection patterns and remote endpoints")
            steps.append("- Investigate associated processes")
            steps.append("- Review firewall and security group configurations")
        elif category == "kernel_params":
            steps.append("- Verify kernel parameter settings against hardening benchmarks")
            steps.append("- Check for related kernel modules")
            steps.append("- Review system logs for exploitation attempts")
        else:
            steps.append("- Cross-reference with other security findings")
            steps.append("- Review system logs for related events")
            steps.append("- Validate against security baselines")
        
        return "\n".join(steps)
    
    def _extract_or_generate_actions(self, finding: Dict[str, Any]) -> str:
        """Extract or generate remediation actions."""
        severity = finding.get("severity", "unknown")
        category = finding.get("metadata", {}).get("scanner_type", "unknown")
        title = finding.get("title", "")
        
        actions = []
        
        # Generate context-appropriate actions
        if "suid" in title.lower() or category == "suid":
            actions.extend([
                "1. **Verify SUID necessity**: Confirm if the SUID permission is required for legitimate functionality",
                "2. **Remove if unnecessary**: `sudo chmod u-s /path/to/binary` to remove SUID bit",
                "3. **Implement capabilities**: Consider using Linux capabilities instead of SUID if applicable",
                "4. **Monitor**: Add to monitoring for unauthorized SUID changes"
            ])
        elif "network" in category.lower():
            actions.extend([
                "1. **Identify service**: Determine which service is using the network port",
                "2. **Validate necessity**: Confirm if the network service is required",
                "3. **Firewall rules**: Implement restrictive firewall rules if service is necessary",
                "4. **Disable if unnecessary**: Stop and disable the service if not required"
            ])
        elif "kernel" in category.lower():
            actions.extend([
                "1. **Review parameter**: Understand the security implication of the kernel parameter",
                "2. **Update configuration**: Modify `/etc/sysctl.conf` with secure values",
                "3. **Apply changes**: Run `sudo sysctl -p` to apply new parameters",
                "4. **Test and validate**: Ensure system functionality is maintained"
            ])
        else:
            actions.extend([
                f"1. **Assess impact**: Evaluate the security implications of this {severity} severity finding",
                "2. **Research remediation**: Consult security documentation for recommended fixes",
                "3. **Implement fix**: Apply the appropriate remediation based on best practices",
                "4. **Verify resolution**: Confirm the security issue is properly addressed"
            ])
        
        return "\n".join(actions)
    
    def _generate_comprehensive_analysis(self, finding: Dict[str, Any]) -> str:
        """Generate comprehensive analysis combining all aspects."""
        
        # Extract all metadata
        risk_subscores = finding.get("risk_subscores", {})
        probability_actionable = finding.get("probability_actionable", 0.5)
        risk_score = finding.get("risk_score", 50)
        correlation_refs = finding.get("correlation_refs", [])
        
        analysis = f"""**Comprehensive Security Analysis:**

**1. Risk Assessment:**
- Overall Risk Score: {risk_score}/100
- Impact Subscore: {risk_subscores.get('impact', 0.5):.2f} - {self._interpret_subscore(risk_subscores.get('impact', 0.5), 'impact')}
- Exposure Subscore: {risk_subscores.get('exposure', 0.5):.2f} - {self._interpret_subscore(risk_subscores.get('exposure', 0.5), 'exposure')}
- Anomaly Subscore: {risk_subscores.get('anomaly', 0.5):.2f} - {self._interpret_subscore(risk_subscores.get('anomaly', 0.5), 'anomaly')}
- Confidence Subscore: {risk_subscores.get('confidence', 0.5):.2f} - {self._interpret_subscore(risk_subscores.get('confidence', 0.5), 'confidence')}

**2. Correlation Analysis:**
{f"Detected {len(correlation_refs)} correlated finding(s) - suggests related security issues" if correlation_refs else "No direct correlations detected - appears to be isolated finding"}

**3. Actionability Assessment:**
Probability Actionable: {probability_actionable:.2f} - {"Immediate action recommended" if probability_actionable > 0.7 else "Schedule remediation" if probability_actionable > 0.4 else "Monitor and assess"}

**4. Remediation Recommendations:**
{self._extract_or_generate_actions(finding)}

**5. Summary:**
{self._generate_risk_reasoning(finding, risk_subscores)}"""
        
        return analysis
    
    def _interpret_subscore(self, score: float, subscore_type: str) -> str:
        """Interpret subscore value."""
        if score > 0.8:
            return f"Very high {subscore_type}"
        elif score > 0.6:
            return f"High {subscore_type}"
        elif score > 0.4:
            return f"Moderate {subscore_type}"
        elif score > 0.2:
            return f"Low {subscore_type}"
        else:
            return f"Very low {subscore_type}"
    
    def preprocess_dataset(
        self,
        input_path: str = "massive_datasets/",
        output_path: str = "structured_dataset",
        max_samples: Optional[int] = None
    ) -> DatasetDict:
        """
        Process synthetic data into structured messages format.
        
        Args:
            input_path: Path to compressed JSON batch files
            output_path: Path to save processed dataset
            max_samples: Optional limit on number of samples (for testing)
        
        Returns:
            DatasetDict with train/validation splits
        """
        
        print("🚀 Starting structured preprocessing...")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        structured_samples = []
        total_findings = 0
        
        # Find all batch files (search recursively in subdirectories)
        input_path_obj = Path(input_path)
        batch_files = sorted(input_path_obj.glob("batch_*.json"))
        
        # If no files found in root, search in subdirectories (e.g., massive_datasets/massive_datasets_max/)
        if not batch_files:
            batch_files = sorted(input_path_obj.glob("**/batch_*.json"))
        
        if not batch_files:
            # Provide helpful error message
            print(f"\n❌ ERROR: No batch files found in {input_path}")
            print(f"\nSearched for:")
            print(f"  - {input_path}/batch_*.json")
            print(f"  - {input_path}/**/batch_*.json (recursive)")
            print(f"\nDirectory contents:")
            if input_path_obj.exists():
                for item in input_path_obj.iterdir():
                    print(f"  - {item.name}{'/' if item.is_dir() else ''}")
            else:
                print(f"  Directory does not exist: {input_path}")
            raise FileNotFoundError(f"No batch files found in {input_path}")
        
        print(f"Found {len(batch_files)} batch files to process")
        
        # Process each batch file
        for batch_file in tqdm(batch_files, desc="Processing batches"):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                # Handle compressed data
                if content.get('compressed'):
                    hex_string = content['data']
                    compressed_bytes = bytes.fromhex(hex_string)
                    decompressed_json_string = gzip.decompress(compressed_bytes).decode('utf-8')
                    decompressed_data = json.loads(decompressed_json_string)
                    
                    if 'data' in decompressed_data and 'findings' in decompressed_data['data']:
                        findings_by_category = decompressed_data['data']['findings']
                        
                        for category, severity_levels in findings_by_category.items():
                            if isinstance(severity_levels, dict):
                                for severity, findings_list in severity_levels.items():
                                    if isinstance(findings_list, list):
                                        for finding in findings_list:
                                            # Sample task type based on distribution
                                            task_type = random.choices(
                                                self.task_types,
                                                weights=self.task_weights
                                            )[0]
                                            
                                            # Create structured messages
                                            try:
                                                structured_sample = self.create_messages_format(
                                                    finding, task_type
                                                )
                                                structured_samples.append(structured_sample)
                                                total_findings += 1
                                                
                                                # Check max_samples limit
                                                if max_samples and total_findings >= max_samples:
                                                    print(f"\n✅ Reached max_samples limit: {max_samples}")
                                                    break
                                            except Exception as e:
                                                print(f"\nError creating messages for finding: {e}")
                                                continue
                                    
                                    if max_samples and total_findings >= max_samples:
                                        break
                            
                            if max_samples and total_findings >= max_samples:
                                break
                
                if max_samples and total_findings >= max_samples:
                    break
                    
            except Exception as e:
                print(f"\nError processing {batch_file}: {e}")
                continue
        
        if not structured_samples:
            raise ValueError("No samples were created from the input files")
        
        print(f"\n✅ Created {len(structured_samples)} structured training samples")
        
        # Task distribution statistics
        task_counts = {}
        for sample in structured_samples:
            user_content = sample['messages'][1]['content']
            for task_type in self.task_types:
                if task_type.replace('_', ' ').title() in user_content or \
                   any(keyword in user_content.lower() for keyword in task_type.split('_')):
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                    break
        
        print("\n📊 Task Distribution:")
        for task_type, count in task_counts.items():
            percentage = (count / len(structured_samples)) * 100
            print(f"  - {task_type}: {count} ({percentage:.1f}%)")
        
        # Create HuggingFace Dataset
        print("\n📦 Creating HuggingFace Dataset...")
        full_dataset = Dataset.from_list(structured_samples)
        
        # Train/validation split (80/20)
        print("🔀 Creating train/validation split (80/20)...")
        train_test_split = full_dataset.train_test_split(test_size=0.2, seed=self.seed)
        
        split_dataset = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  - Train samples: {len(split_dataset['train']):,}")
        print(f"  - Validation samples: {len(split_dataset['validation']):,}")
        
        # Calculate token statistics (approximate)
        sample_msgs = structured_samples[0]['messages']
        avg_user_tokens = len(sample_msgs[1]['content'].split()) * 1.3  # Rough token estimate
        avg_assistant_tokens = len(sample_msgs[2]['content'].split()) * 1.3
        avg_total_tokens = avg_user_tokens + avg_assistant_tokens
        
        print(f"\n📏 Token Statistics (approximate):")
        print(f"  - Avg user prompt tokens: ~{int(avg_user_tokens)}")
        print(f"  - Avg assistant response tokens: ~{int(avg_assistant_tokens)}")
        print(f"  - Avg total tokens per sample: ~{int(avg_total_tokens)}")
        print(f"  - Recommended max_seq_length: {min(2048, int(avg_total_tokens * 1.5))}")
        
        # Save dataset
        print(f"\n💾 Saving dataset to {output_path}...")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        split_dataset.save_to_disk(output_path)
        
        # Save a sample for inspection
        sample_output = f"{output_path}_sample.json"
        with open(sample_output, 'w') as f:
            json.dump(structured_samples[:5], f, indent=2)
        print(f"📄 Saved 5 sample examples to {sample_output}")
        
        print(f"\n🎉 SUCCESS! Structured dataset saved to {output_path}")
        
        return split_dataset


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess synthetic data with structured messages format")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./massive_datasets_max",
        help="Path to directory containing batch_*.json files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./structured_dataset",
        help="Path to save processed dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = StructuredPreprocessor(seed=args.seed)
    
    # Process dataset
    try:
        dataset = preprocessor.preprocess_dataset(
            input_path=args.input_path,
            output_path=args.output_path,
            max_samples=args.max_samples
        )
        
        print("\n✅ Preprocessing complete!")
        print(f"   Dataset ready for training with structured messages format")
        print(f"   Use with SFTTrainer by loading: Dataset.load_from_disk('{args.output_path}')")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()

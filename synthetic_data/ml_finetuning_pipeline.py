"""
ML Fine-tuning Pipeline for Sys-Scan Embedded Model
====================================================

This pipeline orchestrates:
1. Data loading and preprocessing from synthetic security findings
2. Tokenization with proper formatting for security analysis tasks
3. Teacher-student distillation with TRL
4. Training with Lion optimizer
5. Model evaluation and export

Architecture:
- Teacher: Larger pretrained model (e.g., Mistral-7B-Instruct or Qwen-2.5-7B-Instruct)
- Student: 4-bit quantized Mistral-7B with LoRA adapters
- Training: SFT (Supervised Fine-Tuning) with KL divergence distillation
- Optimizer: Lion (EvoLved Sign Momentum)

License: Apache 2.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning pipeline."""

    # Model configurations
    teacher_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    student_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # LoRA configurations
    lora_r: int = 64  # Rank
    lora_alpha: int = 128  # Alpha for scaling
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Lion optimizer specific
    use_lion: bool = True
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99

    # Distillation
    use_distillation: bool = True
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Balance between student loss and distillation

    # Data processing
    max_seq_length: int = 2048
    dataset_text_field: str = "text"

    # Paths
    output_dir: str = "./mistral-7b-security-finetuned"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"

    # Miscellaneous
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"  # Will be overridden if use_lion=True
    save_strategy: str = "epoch"
    logging_steps: int = 10
    eval_steps: int = 50
    save_total_limit: int = 2


class SecurityDataPreprocessor:
    """Preprocesses synthetic security findings for LLM fine-tuning."""

    SYSTEM_PROMPT = """You are a security analysis AI embedded in sys-scan-graph, a system security scanner. Your role is to analyze security findings from various scanners (processes, network, kernel, SUID binaries, IOCs, etc.) and provide:

1. Risk scoring and enrichment
2. Correlation analysis between findings
3. Executive summaries and triage recommendations
4. Actionable remediation steps

You must output valid JSON following the ground truth schema. Be precise, concise, and security-focused."""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize preprocessor with optional schema validation."""
        self.schema = None
        if schema_path and Path(schema_path).exists():
            with open(schema_path) as f:
                self.schema = json.load(f)

    def load_synthetic_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load synthetic security findings from JSON."""
        logger.info(f"Loading synthetic data from {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        # Handle both nested and flat structures
        if 'data' in data:
            findings = data['data']
        else:
            findings = data

        logger.info(f"Loaded {len(findings.get('findings', {}))} finding categories")
        return findings

    def create_training_examples(
        self,
        findings_data: Dict[str, Any],
        format_type: str = "completion"
    ) -> List[Dict[str, str]]:
        """
        Convert security findings into training examples.

        Args:
            findings_data: Raw findings from synthetic data generator
            format_type: "completion" for instruction-following or "chat" for chat format

        Returns:
            List of formatted training examples
        """
        examples = []

        # Extract findings by category
        findings_by_category = findings_data.get('findings', {})

        # Create examples for different analysis tasks
        examples.extend(self._create_risk_scoring_examples(findings_by_category))
        examples.extend(self._create_correlation_examples(findings_data))
        examples.extend(self._create_summary_examples(findings_data))

        logger.info(f"Created {len(examples)} training examples")
        return examples

    def _create_risk_scoring_examples(
        self,
        findings_by_category: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create examples for risk scoring and enrichment tasks."""
        examples = []

        for category, severity_groups in findings_by_category.items():
            if not isinstance(severity_groups, dict):
                continue

            for severity, findings_list in severity_groups.items():
                if not isinstance(findings_list, list):
                    continue

                for finding in findings_list[:5]:  # Limit per category
                    # Input: raw finding
                    input_finding = {
                        "id": finding.get("id"),
                        "title": finding.get("title"),
                        "description": finding.get("description"),
                        "metadata": finding.get("metadata", {}),
                        "category": category
                    }

                    # Output: enriched finding with risk scores
                    output_finding = {
                        k: v for k, v in finding.items()
                        if k in [
                            "risk_score", "risk_subscores", "severity",
                            "probability_actionable", "baseline_status",
                            "tags", "rationale"
                        ]
                    }

                    instruction = "Analyze this security finding and provide risk scoring with subscores (impact, exposure, anomaly, confidence), severity classification, and actionability assessment."

                    text = self._format_example(
                        instruction=instruction,
                        input_data=input_finding,
                        output_data=output_finding
                    )

                    examples.append({"text": text})

        return examples

    def _create_correlation_examples(
        self,
        findings_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create examples for correlation analysis tasks."""
        examples = []

        correlations = findings_data.get('correlations', [])
        if not correlations:
            return examples

        # Get all findings for context
        all_findings = []
        findings_by_category = findings_data.get('findings', {})
        for category, severity_groups in findings_by_category.items():
            if isinstance(severity_groups, dict):
                for findings_list in severity_groups.values():
                    if isinstance(findings_list, list):
                        all_findings.extend(findings_list)

        for corr in correlations[:10]:  # Limit correlations
            related_ids = corr.get('related_finding_ids', [])
            if not related_ids:
                continue

            # Get related findings
            related_findings = [
                f for f in all_findings
                if f.get('id') in related_ids
            ]

            if len(related_findings) < 2:
                continue

            input_data = {
                "findings": [
                    {
                        "id": f.get("id"),
                        "title": f.get("title"),
                        "category": f.get("category"),
                        "severity": f.get("severity"),
                        "tags": f.get("tags", [])
                    }
                    for f in related_findings
                ]
            }

            output_data = {
                "correlation_id": corr.get('id'),
                "title": corr.get('title'),
                "rationale": corr.get('rationale'),
                "severity": corr.get('severity'),
                "related_finding_ids": related_ids
            }

            instruction = "Identify correlations between these security findings. Provide a correlation title, rationale explaining the relationship, and severity assessment."

            text = self._format_example(
                instruction=instruction,
                input_data=input_data,
                output_data=output_data
            )

            examples.append({"text": text})

        return examples

    def _create_summary_examples(
        self,
        findings_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create examples for summary generation tasks."""
        examples = []

        # Get reductions and summaries if available
        reductions = findings_data.get('reductions', {})
        summaries = findings_data.get('summaries', {})

        if not (reductions and summaries):
            return examples

        # Executive summary example
        if 'executive_summary' in summaries:
            input_data = {
                "top_findings": reductions.get('top_findings', [])[:5],
                "module_summary": reductions.get('module_summary'),
                "network_summary": reductions.get('network_summary')
            }

            output_data = {
                "executive_summary": summaries['executive_summary']
            }

            instruction = "Generate a concise executive summary of these security findings for leadership review."

            text = self._format_example(
                instruction=instruction,
                input_data=input_data,
                output_data=output_data
            )

            examples.append({"text": text})

        return examples

    def _format_example(
        self,
        instruction: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any]
    ) -> str:
        """Format example in instruction-following format (Mistral style)."""
        # Using Mistral-Instruct chat template format
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nInput:\n{json.dumps(input_data, indent=2)}"},
            {"role": "assistant", "content": json.dumps(output_data, indent=2)}
        ]

        # Format as Mistral chat template
        formatted = "<s>"
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"[INST] {msg['content']} [/INST]\n"
            elif msg["role"] == "user":
                formatted += f"[INST] {msg['content']} [/INST]\n"
            elif msg["role"] == "assistant":
                formatted += f"{msg['content']}</s>\n"

        return formatted


class SecurityFineTuningPipeline:
    """Complete fine-tuning pipeline for security analysis model."""

    def __init__(self, config: FinetuningConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.tokenizer = None
        self.teacher_model = None
        self.student_model = None
        self.preprocessor = SecurityDataPreprocessor()

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def setup_tokenizer(self) -> AutoTokenizer:
        """Setup tokenizer with proper padding configuration."""
        logger.info(f"Loading tokenizer from {self.config.student_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir
        )

        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer
        return tokenizer

    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization configuration."""
        if not self.config.use_4bit:
            return None

        logger.info("Configuring 4-bit quantization (NF4)")

        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )

    def setup_models(self) -> Tuple[Any, Any]:
        """Setup teacher and student models with quantization and LoRA."""
        bnb_config = self.setup_quantization_config()

        # Load student model
        logger.info(f"Loading student model: {self.config.student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )

        # Prepare for k-bit training
        student_model = prepare_model_for_kbit_training(
            student_model,
            use_gradient_checkpointing=self.config.gradient_checkpointing
        )

        # Setup LoRA
        logger.info(f"Applying LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        student_model = get_peft_model(student_model, peft_config)
        student_model.print_trainable_parameters()

        # Optionally load teacher model for distillation
        teacher_model = None
        if self.config.use_distillation:
            logger.info(f"Loading teacher model: {self.config.teacher_model_name}")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model_name,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
            teacher_model.eval()  # Teacher in eval mode

        self.student_model = student_model
        self.teacher_model = teacher_model

        return teacher_model, student_model

    def prepare_dataset(
        self,
        data_paths: List[str],
        train_split: float = 0.9
    ) -> DatasetDict:
        """
        Prepare training and evaluation datasets.

        Args:
            data_paths: List of paths to synthetic data JSON files
            train_split: Fraction of data to use for training

        Returns:
            DatasetDict with train and eval splits
        """
        logger.info(f"Preparing datasets from {len(data_paths)} files")

        all_examples = []

        for data_path in data_paths:
            findings_data = self.preprocessor.load_synthetic_data(data_path)
            examples = self.preprocessor.create_training_examples(findings_data)
            all_examples.extend(examples)

        logger.info(f"Total training examples: {len(all_examples)}")

        # Shuffle and split
        np.random.shuffle(all_examples)
        split_idx = int(len(all_examples) * train_split)

        train_data = all_examples[:split_idx]
        eval_data = all_examples[split_idx:]

        # Create HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'eval': Dataset.from_list(eval_data)
        })

        logger.info(f"Train examples: {len(train_data)}, Eval examples: {len(eval_data)}")

        return dataset_dict

    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments with Lion optimizer if configured."""
        logger.info("Configuring training arguments")

        # Override optimizer if using Lion
        optim = self.config.optim
        optim_args = {}

        if self.config.use_lion:
            logger.info("Using Lion optimizer")
            # TRL/Transformers supports custom optimizers via optim string
            # For Lion, we'll need to use a custom approach or use AdamW as fallback
            # Since Lion isn't natively supported, we'll configure it in trainer
            optim = "adamw_torch"  # Fallback, will customize in trainer
            optim_args = {
                "betas": (self.config.lion_beta1, self.config.lion_beta2)
            }

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            optim=optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        return training_args

    def train(
        self,
        dataset: DatasetDict,
        training_args: TrainingArguments
    ) -> SFTTrainer:
        """
        Execute training with SFTTrainer from TRL.

        Args:
            dataset: Prepared dataset with train/eval splits
            training_args: Training configuration

        Returns:
            Trained SFTTrainer instance
        """
        logger.info("Initializing SFTTrainer")

        # Data collator for completion-only training
        # This masks the instruction part, only computing loss on responses
        response_template = "[/INST]"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = SFTTrainer(
            model=self.student_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            tokenizer=self.tokenizer,
            dataset_text_field=self.config.dataset_text_field,
            max_seq_length=self.config.max_seq_length,
            data_collator=collator,
            packing=False,  # Don't pack multiple examples together
        )

        # TODO: Add custom Lion optimizer if needed
        # This would require subclassing SFTTrainer and overriding create_optimizer

        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed!")
        return trainer

    def save_model(self, trainer: SFTTrainer, output_path: Optional[str] = None):
        """Save fine-tuned model and tokenizer."""
        save_path = output_path or self.config.output_dir

        logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info("Model saved successfully!")

    def run_pipeline(self, data_paths: List[str]) -> SFTTrainer:
        """
        Run complete fine-tuning pipeline.

        Args:
            data_paths: List of paths to synthetic data files

        Returns:
            Trained SFTTrainer instance
        """
        logger.info("="*60)
        logger.info("Starting Security Model Fine-Tuning Pipeline")
        logger.info("="*60)

        # Setup
        self.setup_tokenizer()
        self.setup_models()

        # Prepare data
        dataset = self.prepare_dataset(data_paths)

        # Training
        training_args = self.setup_training_args()
        trainer = self.train(dataset, training_args)

        # Save
        self.save_model(trainer)

        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)

        return trainer


def create_default_pipeline(
    output_dir: str = "./mistral-7b-security-finetuned",
    use_distillation: bool = False
) -> SecurityFineTuningPipeline:
    """Create pipeline with default configuration."""
    config = FinetuningConfig(
        output_dir=output_dir,
        use_distillation=use_distillation,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    )

    return SecurityFineTuningPipeline(config)


if __name__ == "__main__":
    # Example usage
    pipeline = create_default_pipeline()

    # Provide paths to your synthetic data
    data_paths = [
        "/home/joseph-mazzini/sys-scan-embedded-agent/synthetic_data/test_pipeline_output.json"
    ]

    trainer = pipeline.run_pipeline(data_paths)

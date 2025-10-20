"""
Knowledge Distillation Trainer for Security Model
==================================================

Implements teacher-student distillation for transferring knowledge from
a larger teacher model to a smaller, quantized student model.

Combines:
- Standard cross-entropy loss (student vs ground truth)
- KL divergence loss (student vs teacher logits)
- Optional intermediate layer matching

License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Combines:
    1. Student loss: Cross-entropy with ground truth labels
    2. Distillation loss: KL divergence between student and teacher predictions
    3. Optional cosine similarity between hidden states
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_hidden_states: bool = False,
        hidden_loss_weight: float = 0.1
    ):
        """
        Args:
            temperature: Temperature for softmax in distillation
            alpha: Weight for distillation loss (1-alpha for student loss)
            use_hidden_states: Whether to match hidden states
            hidden_loss_weight: Weight for hidden state matching loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_hidden_states = use_hidden_states
        self.hidden_loss_weight = hidden_loss_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab]
            teacher_logits: Teacher model logits [batch, seq_len, vocab]
            labels: Ground truth labels [batch, seq_len]
            student_hidden: Student hidden states (optional)
            teacher_hidden: Teacher hidden states (optional)

        Returns:
            Total loss and dictionary of individual loss components
        """
        # 1. Student loss (cross-entropy with labels)
        student_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 2. Distillation loss (KL divergence with teacher)
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence loss
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False
        ) * (self.temperature ** 2)

        # 3. Combined loss
        total_loss = (
            self.alpha * distillation_loss +
            (1 - self.alpha) * student_loss
        )

        loss_dict = {
            "student_loss": student_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "total_loss": total_loss.item()
        }

        # 4. Optional hidden state matching
        if self.use_hidden_states and student_hidden is not None and teacher_hidden is not None:
            # Cosine similarity loss between hidden states
            hidden_loss = 1 - F.cosine_similarity(
                student_hidden.mean(dim=1),  # Average over sequence
                teacher_hidden.mean(dim=1),
                dim=-1
            ).mean()

            total_loss = total_loss + self.hidden_loss_weight * hidden_loss
            loss_dict["hidden_loss"] = hidden_loss.item()

        return total_loss, loss_dict


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Extends HuggingFace Trainer to support teacher-student training.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_hidden_states: bool = False,
        *args,
        **kwargs
    ):
        """
        Args:
            teacher_model: Pre-trained teacher model (frozen)
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            use_hidden_states: Whether to match hidden states
            *args, **kwargs: Arguments for base Trainer
        """
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Teacher always in eval mode

        # Move teacher to same device as student
        if self.args.device is not None:
            self.teacher_model.to(self.args.device)

        self.distillation_loss_fn = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            use_hidden_states=use_hidden_states
        )

        # Track distillation metrics
        self.distillation_metrics = {
            "student_loss": [],
            "distillation_loss": [],
            "hidden_loss": []
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            model: Student model
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Get student outputs
        student_outputs = model(**inputs, output_hidden_states=self.distillation_loss_fn.use_hidden_states)
        student_logits = student_outputs.logits

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs,
                output_hidden_states=self.distillation_loss_fn.use_hidden_states
            )
            teacher_logits = teacher_outputs.logits

        # Extract hidden states if needed
        student_hidden = None
        teacher_hidden = None
        if self.distillation_loss_fn.use_hidden_states:
            student_hidden = student_outputs.hidden_states[-1]  # Last layer
            teacher_hidden = teacher_outputs.hidden_states[-1]

        # Compute distillation loss
        loss, loss_dict = self.distillation_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=inputs["labels"],
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden
        )

        # Track metrics
        for key, value in loss_dict.items():
            if key in self.distillation_metrics:
                self.distillation_metrics[key].append(value)

        # Log periodically
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_distillation_metrics()

        if return_outputs:
            return loss, student_outputs

        return loss

    def _log_distillation_metrics(self):
        """Log average distillation metrics."""
        if not self.distillation_metrics["student_loss"]:
            return

        avg_metrics = {}
        for key, values in self.distillation_metrics.items():
            if values:
                avg_metrics[f"train/{key}"] = sum(values) / len(values)

        self.log(avg_metrics)

        # Clear metrics
        for key in self.distillation_metrics:
            self.distillation_metrics[key] = []

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save checkpoint and distillation config."""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # Save model checkpoint
        output = super()._save_checkpoint(model, trial, metrics=metrics)

        # Save distillation config
        if self.args.should_save:
            import json
            from pathlib import Path

            distill_config = {
                "temperature": self.distillation_loss_fn.temperature,
                "alpha": self.distillation_loss_fn.alpha,
                "use_hidden_states": self.distillation_loss_fn.use_hidden_states,
                "hidden_loss_weight": self.distillation_loss_fn.hidden_loss_weight,
            }

            config_path = Path(self.args.output_dir) / checkpoint_folder / "distillation_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                json.dump(distill_config, f, indent=2)

        return output

    def evaluation_loop(self, *args, **kwargs):
        """Run evaluation with teacher model on same device."""
        # Ensure teacher is on correct device
        if self.args.device is not None and next(self.teacher_model.parameters()).device != self.args.device:
            self.teacher_model.to(self.args.device)

        return super().evaluation_loop(*args, **kwargs)


def create_distillation_trainer(
    student_model: nn.Module,
    teacher_model: nn.Module,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator=None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    use_hidden_states: bool = False,
) -> DistillationTrainer:
    """
    Create a configured DistillationTrainer.

    Args:
        student_model: Model to train
        teacher_model: Pre-trained teacher model
        training_args: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        data_collator: Optional data collator
        temperature: Distillation temperature
        alpha: Distillation loss weight
        use_hidden_states: Match hidden states

    Returns:
        Configured DistillationTrainer
    """
    logger.info("Creating DistillationTrainer")
    logger.info(f"Temperature: {temperature}, Alpha: {alpha}")

    return DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        temperature=temperature,
        alpha=alpha,
        use_hidden_states=use_hidden_states,
    )


if __name__ == "__main__":
    # Test distillation loss
    batch_size, seq_len, vocab_size = 2, 10, 1000

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
    loss, loss_dict = loss_fn(student_logits, teacher_logits, labels)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    print("Distillation loss test passed!")

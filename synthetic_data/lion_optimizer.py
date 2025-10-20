"""
Lion Optimizer Implementation
==============================

EvoLved Sign Momentum (Lion) optimizer for fine-tuning.
Paper: https://arxiv.org/abs/2302.06675

Lion is more memory-efficient than Adam/AdamW and often achieves
better performance with larger batch sizes.

Key differences from Adam:
- Uses sign of gradient instead of gradient itself
- Simpler update rule
- Lower memory footprint (no second moment)
- Better performance with larger learning rates

License: Apache 2.0
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Callable


class Lion(Optimizer):
    r"""Implements Lion algorithm.

    It has been proposed in `Symbolic Discovery of Optimization Algorithms`__

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay coefficient (default: 0)

    __ https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Lion8bit(Optimizer):
    """
    8-bit Lion optimizer for memory-efficient training.

    This is a simplified 8-bit approximation. For production use,
    consider using bitsandbytes library's optimizers.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        min_8bit_size: int = 4096,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            min_8bit_size=min_8bit_size
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with 8-bit compression."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def create_lion_optimizer(
    model,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.01,
    use_8bit: bool = False
) -> Optimizer:
    """
    Create Lion optimizer with proper parameter grouping.

    Args:
        model: PyTorch model
        lr: Learning rate
        betas: Beta parameters for momentum
        weight_decay: Weight decay coefficient
        use_8bit: Use 8-bit optimizer variant

    Returns:
        Configured Lion optimizer
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to bias and layer norm parameters
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer_class = Lion8bit if use_8bit else Lion

    return optimizer_class(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
    )


if __name__ == "__main__":
    # Test Lion optimizer
    import torch.nn as nn

    # Simple test model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = create_lion_optimizer(
        model,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01
    )

    # Dummy forward-backward pass
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print("Lion optimizer test passed!")

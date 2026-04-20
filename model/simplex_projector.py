"""
Step 5: Projected Gradient Descent onto the probability simplex.

Implements the Duchi et al. (2008) O(n log n) exact projection algorithm.
This is used as a post-processing step after each TFT backward pass to
enforce:
  (1) all weights >= 0  (long-only constraint)
  (2) sum of weights == 1  (fully invested constraint)

Softmax cannot produce true zeros; PGD onto the simplex can, allowing the
network to learn genuine position exits rather than infinitesimally small
residuals.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def project_onto_simplex(v: Tensor) -> Tensor:
    """
    Project each row of v onto the unit probability simplex.
    Supports batched input: v has shape (..., n_assets).

    Algorithm: Duchi et al. (2008) "Efficient Projections onto the L1-Ball
    for Learning in High Dimensions."  O(n log n) per vector.
    """
    shape = v.shape
    n = shape[-1]
    # Flatten all batch dims into one
    u = v.reshape(-1, n)

    # Sort descending
    u_sorted, _ = torch.sort(u, dim=-1, descending=True)
    cssv = torch.cumsum(u_sorted, dim=-1)
    rho_candidates = u_sorted - (cssv - 1.0) / (
        torch.arange(1, n + 1, dtype=v.dtype, device=v.device).unsqueeze(0)
    )
    # rho is the last index where the condition holds
    rho = (rho_candidates > 0).sum(dim=-1) - 1  # (batch,)
    rho = rho.clamp(min=0)

    theta = (cssv.gather(1, rho.unsqueeze(1)) - 1.0) / (rho + 1).float().unsqueeze(1)
    projected = (u - theta).clamp(min=0.0)

    # Re-normalise for floating-point safety
    row_sums = projected.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    projected = projected / row_sums

    return projected.reshape(shape)


class SimplexProjectionLayer(nn.Module):
    """
    A differentiable simplex projection layer that can be inserted at the
    output of the TFT decoder.

    Note: The projection is differentiable almost everywhere; the gradient
    flows through the non-zero components as an identity.
    """

    def forward(self, x: Tensor) -> Tensor:
        return project_onto_simplex(x)


class ProjectedGradientOptimizer:
    """
    Wraps any PyTorch optimizer and applies simplex projection to a
    designated weight vector parameter after each step.

    Usage:
        base_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        opt = ProjectedGradientOptimizer(base_opt, weight_param=model.portfolio_weights)
        opt.step()
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, weight_param: nn.Parameter):
        self.base_optimizer = base_optimizer
        self.weight_param = weight_param

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def step(self) -> None:
        self.base_optimizer.step()
        with torch.no_grad():
            self.weight_param.data = project_onto_simplex(self.weight_param.data)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state):
        self.base_optimizer.load_state_dict(state)


def tracking_error_loss(predicted_weights: Tensor, actual_weights: Tensor) -> Tensor:
    """
    Minimize tracking error between predicted and realised 13F weights.
    TE = sqrt(mean((w_pred - w_actual)^2))

    predicted_weights: (batch, n_assets)
    actual_weights:    (batch, n_assets)
    """
    diff = predicted_weights - actual_weights
    return torch.sqrt((diff ** 2).mean(dim=-1)).mean()


def portfolio_entropy_regularizer(weights: Tensor, lambda_: float = 0.01) -> Tensor:
    """
    Entropy regularizer to discourage extreme concentration.
    Encourages the model to maintain some diversification baseline.
    H = -sum(w * log(w + eps))
    Loss term = -lambda * H  (negative because we maximise entropy)
    """
    eps = 1e-9
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
    return -lambda_ * entropy

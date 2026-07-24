"""Gaussian likelihood guidance target ``y = A x + N(0, σ²)`` — a soft tilt for inverse problems.

log_h = −‖A x − y‖² / (2σ²). Used by `Guidance` (no hard projection). Affine push-through exact.
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Cost
from forge.core.registry import register


@register("cost", "likelihood")
class Likelihood(Cost):
    def __init__(self, A, y, sigma: float = 0.1):
        self.A = torch.as_tensor(A, dtype=torch.float32)        # (m, D)
        self.y = torch.as_tensor(y, dtype=torch.float32)        # (m,)
        self.sigma = float(sigma)

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        A, y = self.A.to(x.device), self.y.to(x.device)
        resid = x @ A.T - y                                     # (B, m)
        return -0.5 * (resid**2).sum(dim=-1) / (self.sigma**2)

    def to_normalized(self, preprocessor) -> "Likelihood":
        scale = preprocessor.scale().to(self.A.device)          # (D,)
        shift = preprocessor.shift().to(self.A.device)
        # A x = A (x̃ − shift)/scale = (A/scale) x̃ − A·(shift/scale)
        A_n = self.A / scale
        y_n = self.y + (self.A * (shift / scale)).sum(dim=-1)
        return Likelihood(A_n, y_n, self.sigma)

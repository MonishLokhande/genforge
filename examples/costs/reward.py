"""Reward guidance target — a smooth quadratic bowl ``log_h = −w‖x − target‖²``.

A differentiable everywhere guidance target (unlike a hard constraint); ∇log_h points toward the
target. Used online by `Guidance`, and as the reward a `value_training` method amortizes.
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Cost
from forge.core.registry import register


@register("cost", "reward")
class Reward(Cost):
    """``log_h = -Σ_i w_i (x_i − target_i)²``. ``weight`` is scalar or per-feature; the per-feature
    form is what makes the affine push-through exact under an anisotropic membrane (Invariant 8):
    the physical squared distance carries a diagonal metric ``1/scale_i²`` that a single scalar
    cannot represent."""

    def __init__(self, target, weight=1.0):
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.weight = torch.as_tensor(weight, dtype=torch.float32)  # scalar or (dim,)

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        w = self.weight.to(x.device)
        return -(w * (x - self.target.to(x.device)) ** 2).sum(dim=-1)

    def to_normalized(self, preprocessor) -> "Reward":
        # x = (x̃ − shift)/scale  ⟹  (x_i − target_i)² = (x̃_i − target̃_i)²/scale_i²,
        # so the target maps affinely and the per-feature weight picks up 1/scale_i² (EXACT).
        scale = preprocessor.scale().to(self.target.device)
        shift = preprocessor.shift().to(self.target.device)
        return Reward(self.target * scale + shift, self.weight / scale**2)

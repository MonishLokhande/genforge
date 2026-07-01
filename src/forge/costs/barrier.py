"""CBF barrier cost — a function ``h(x)`` whose feasible set is ``h(x) ≥ 0``, enforced along the
sampling path as ``ḣ(x) ≥ −α·h(x)`` (a control barrier function).

A linear barrier ``h(x) = aᵀx + b`` (so ``∇h = a`` is constant). Exposes ``value``/``grad_h`` for the
CBF drift filter; ``log_h`` returns the barrier value (a soft tilt). Affine ``to_normalized`` is
per-feature-exact and preserves the barrier value, so the CBF margin is identical in real and
normalized coordinates (Invariant 8).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Cost
from ..core.registry import register


@register("cost", "barrier")
class Barrier(Cost):
    def __init__(self, normal, offset: float = 0.0):
        self.a = torch.as_tensor(normal, dtype=torch.float32)
        self.b = float(offset)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a.to(x.device) + self.b                # h(x), (...,)

    def grad_h(self, x: torch.Tensor) -> torch.Tensor:
        return self.a.to(x.device).expand_as(x)                # ∇h = a, (..., dim)

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return self.value(x)

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) >= 0

    def to_normalized(self, preprocessor) -> "Barrier":
        # h(x)=aᵀx+b with x=(x̃−shift)/scale  ⟹  h̃(x̃)=(a/scale)ᵀx̃ + (b − Σ a·shift/scale) ≡ h(x).
        scale = preprocessor.scale().to(self.a.device)
        shift = preprocessor.shift().to(self.a.device)
        return Barrier(self.a / scale, self.b - float((self.a * shift / scale).sum().item()))

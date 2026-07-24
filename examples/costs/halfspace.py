"""Halfspace constraint ``aᵀx ≥ b`` (authored in real units).

`log_h` is a hinge log-density (0 when feasible, linear penalty when not) — its gradient pushes a
point toward the feasible side. `project` is the exact closed-form projection onto the halfspace.
`to_normalized` is exact for affine preprocessors (Invariant 8).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forge.core.interfaces import Cost
from forge.core.registry import register


@register("cost", "halfspace")
class Halfspace(Cost):
    def __init__(self, normal, offset: float = 0.0, weight: float = 1.0):
        self.a = torch.as_tensor(normal, dtype=torch.float32)
        self.b = float(offset)
        self.weight = float(weight)

    def _margin(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a.to(x.device) - self.b               # (B,)  ≥ 0 ⇔ feasible

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return -self.weight * F.relu(-self._margin(x))

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        return self._margin(x) >= 0

    def project(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a.to(x.device)
        margin = self._margin(x)                              # (B,)
        # Move infeasible points (margin < 0) onto the boundary along a; leave feasible points.
        step = torch.clamp(-margin, min=0.0) / (a @ a)
        return x + step.unsqueeze(-1) * a

    def to_normalized(self, preprocessor) -> "Halfspace":
        scale = preprocessor.scale().to(self.a.device)
        shift = preprocessor.shift().to(self.a.device)
        a_n = self.a / scale
        b_n = self.b + float((self.a * shift / scale).sum().item())
        return Halfspace(a_n, b_n, self.weight)

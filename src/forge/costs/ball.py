"""Euclidean ball constraint ``‖x − c‖ ≤ r`` (real units).

Push-through is exact only for an *isotropic* affine preprocessor (a non-isotropic scale turns the
ball into an ellipsoid); we transform the center exactly and scale the radius by the mean scale,
and document the caveat.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.interfaces import Cost
from ..core.registry import register


@register("cost", "ball")
class Ball(Cost):
    def __init__(self, center, radius: float, weight: float = 1.0):
        self.center = torch.as_tensor(center, dtype=torch.float32)
        self.radius = float(radius)
        self.weight = float(weight)

    def _dist(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x - self.center.to(x.device), dim=-1)

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return -self.weight * F.relu(self._dist(x) - self.radius)

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist(x) <= self.radius

    def project(self, x: torch.Tensor) -> torch.Tensor:
        c = self.center.to(x.device)
        d = self._dist(x).clamp_min(1e-12)
        scale = torch.clamp(self.radius / d, max=1.0)        # 1 inside, <1 outside
        return c + (x - c) * scale.unsqueeze(-1)

    def to_normalized(self, preprocessor) -> "Ball":
        scale = preprocessor.scale().to(self.center.device)
        shift = preprocessor.shift().to(self.center.device)
        center_n = self.center * scale + shift               # exact (= transform(center))
        radius_n = self.radius * float(scale.mean().item())  # exact iff scale is isotropic
        return Ball(center_n, radius_n, self.weight)

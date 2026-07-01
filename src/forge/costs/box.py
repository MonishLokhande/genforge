"""Axis-aligned box constraint ``lo ≤ x ≤ hi`` (real units). Affine push-through is exact."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.interfaces import Cost
from ..core.registry import register


@register("cost", "box")
class Box(Cost):
    def __init__(self, low, high, weight: float = 1.0):
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.weight = float(weight)

    def log_h(self, x: torch.Tensor, t=None) -> torch.Tensor:
        lo, hi = self.low.to(x.device), self.high.to(x.device)
        penalty = F.relu(lo - x) + F.relu(x - hi)
        return -self.weight * penalty.sum(dim=-1)

    def feasible(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.low.to(x.device), self.high.to(x.device)
        return ((x >= lo) & (x <= hi)).all(dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.low.to(x.device), self.high.to(x.device))

    def to_normalized(self, preprocessor) -> "Box":
        scale = preprocessor.scale().to(self.low.device)   # > 0 for standardize/minmax
        shift = preprocessor.shift().to(self.low.device)
        return Box(self.low * scale + shift, self.high * scale + shift, self.weight)

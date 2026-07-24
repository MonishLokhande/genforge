"""A scalar value/reward model V(x) — the amortized artifact for value-guided control.

`output_type = "value"`. Applied per element so it handles points ``(B, dim)`` and trajectory
windows ``(B, H, dim)`` alike; ∇_x V drives `ValueGuidance` / `FBSDEControl`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from forge.core.interfaces import Model
from forge.core.registry import register


@register("model", "value_mlp")
class ValueMLP(Model):
    output_type = "value"

    def __init__(self, dim: int = 2, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.dim = dim
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, cond=None) -> torch.Tensor:
        self._check_cond(cond)
        return self.net(x)                                    # (..., 1)

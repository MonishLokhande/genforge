"""D3PM objective: x₀-prediction cross-entropy (discrete is always x₀-prediction).

Uses the three primitives only: corrupt with the forward primitive ``space.forward_sample`` (which
samples from the schedule's ``Q̄_t``), predict x₀ logits, and minimize cross-entropy to the clean
tokens. No cont/disc branch — this is a discrete *method*, not a shared one branching.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..core.interfaces import Method, Model
from ..core.registry import register


@register("method", "d3pm")
class D3PM(Method):
    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.t_eps = float(t_eps)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)
        logits = model(xt, t, cond)                      # (..., V)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x0.reshape(-1))

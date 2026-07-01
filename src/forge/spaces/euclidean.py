"""Euclidean state space: a standard-normal prior and the Gaussian forward kernel.

The forward primitive ``x_t = α x_0 + σ ε``. All α/σ math is delegated to the
schedule, so this space carries no schedule-specific knowledge — only that the space is real-valued
and the noise is Gaussian.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.interfaces import Schedule, Space
from ..core.registry import register


@register("space", "euclidean")
class Euclidean(Space):
    def __init__(self, dim: int = 2):
        self.dim = dim

    def prior_sample(
        self,
        shape,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.randn(*tuple(shape), generator=generator, device=device, dtype=dtype)

    def forward_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        schedule: Schedule,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = N(α(t) x_0, σ(t)² I). The forward primitive."""
        mean, std = schedule.marginal(x0, t)
        eps = torch.randn(
            x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
        )
        return mean + std * eps

"""Discrete (categorical) state space.

The prior is the schedule's stationary noise (all-mask for absorbing, uniform otherwise) and the
forward kernel samples ``x_t ~ q(x_t | x_0)`` from the schedule's ``Q̄_t`` row. This is the discrete
half of Invariant 1 — discreteness lives ONLY here and in `schedule`; every downstream component
stays agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core.interfaces import Schedule, Space
from forge.core.registry import register


@register("space", "discrete")
class Discrete(Space):
    def __init__(self, num_classes: int = 5, length: int = 1, mask_index: Optional[int] = None):
        self.num_classes = int(num_classes)
        self.length = int(length)
        # An absorbing space pins the prior on the mask token; a uniform space samples uniformly.
        self.mask_index = mask_index

    @property
    def dim(self) -> int:
        return self.length

    def prior_sample(
        self,
        shape,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        shape = tuple(shape)
        if self.mask_index is not None:
            return torch.full(shape, self.mask_index, dtype=torch.long, device=device)
        return torch.randint(
            0, self.num_classes, shape, generator=generator, device=device, dtype=torch.long
        )

    def forward_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        schedule: Schedule,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        probs = schedule.qt_probs(x0, t)            # (..., V)
        flat = probs.reshape(-1, probs.shape[-1])
        idx = torch.multinomial(flat, 1, generator=generator).reshape(x0.shape)
        return idx.to(torch.long)

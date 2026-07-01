"""A dependency-free discrete target: i.i.d. categorical tokens with a fixed, skewed distribution.

``evaluate`` reports the L1 distance between the generated class frequencies and the true ones —
the discrete analogue of the continuous mode-coverage metric, delegated to the environment so the
runner stays cont/disc-agnostic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from forge.core.registry import register


@register("environment", "categorical_toy")
class CategoricalToy:
    def __init__(
        self,
        num_classes: int = 4,
        length: int = 1,
        probs: Optional[Sequence[float]] = None,
    ):
        self.num_classes = int(num_classes)
        self.length = int(length)
        if probs is None:
            # A skewed default so "recovers the frequencies" is a non-trivial check.
            probs = [(i + 1) for i in range(self.num_classes)]
        p = torch.tensor(probs, dtype=torch.float32)
        self.probs = p / p.sum()

    @property
    def dim(self) -> int:
        return self.length

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        flat = torch.multinomial(
            self.probs, n * self.length, replacement=True, generator=generator
        )
        return flat.reshape(n, self.length).to(torch.long)

    def evaluate(self, samples: torch.Tensor) -> dict:
        counts = torch.bincount(
            samples.reshape(-1), minlength=self.num_classes
        )[: self.num_classes].to(torch.float32)
        emp = counts / counts.sum().clamp_min(1)
        l1 = float((emp - self.probs.to(emp.device)).abs().sum().item())
        return {"freq_l1": l1, "n": float(samples.numel()), "num_classes": float(self.num_classes)}

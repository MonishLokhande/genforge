"""Synthetic 2-D target distributions — the dependency-free playground for the core paradigms.

Each environment exposes ``sample(n, generator)`` returning ``(n, dim)`` real samples and a ``dim``.
These are raw-unit data sources (outside the preprocessor membrane, Invariant 2).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from forge.core.registry import register


@register("environment", "gaussian_mixture")
class GaussianMixture2D:
    """An isotropic Gaussian mixture. Defaults to the bimodal target at (−2,0)/(+2,0)."""

    def __init__(
        self,
        means: Sequence[Sequence[float]] = ((-2.0, 0.0), (2.0, 0.0)),
        std: float = 0.2,
        weights: Optional[Sequence[float]] = None,
    ):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.std = float(std)
        if weights is None:
            weights = [1.0 / len(self.means)] * len(self.means)
        w = torch.tensor(weights, dtype=torch.float32)
        self.weights = w / w.sum()

    @property
    def dim(self) -> int:
        return self.means.shape[-1]

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        comp = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        centers = self.means[comp]
        noise = torch.randn(n, self.dim, generator=generator) * self.std
        return centers + noise


@register("environment", "two_moons")
class TwoMoons:
    """The classic two-interleaving-half-moons target."""

    def __init__(self, noise: float = 0.1, scale: float = 1.5):
        self.noise = float(noise)
        self.scale = float(scale)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        n0 = n // 2
        n1 = n - n0
        t0 = torch.rand(n0, generator=generator) * math.pi
        outer = torch.stack([torch.cos(t0), torch.sin(t0)], dim=-1)
        t1 = torch.rand(n1, generator=generator) * math.pi
        inner = torch.stack([1.0 - torch.cos(t1), 0.5 - torch.sin(t1)], dim=-1)
        x = torch.cat([outer, inner], dim=0) * self.scale
        x = x + torch.randn(n, 2, generator=generator) * self.noise
        return x


@register("environment", "swiss_roll")
class SwissRoll:
    """A 2-D swiss-roll spiral."""

    def __init__(self, noise: float = 0.05, scale: float = 0.2):
        self.noise = float(noise)
        self.scale = float(scale)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=generator))
        x = t * torch.cos(t)
        y = t * torch.sin(t)
        pts = torch.stack([x, y], dim=-1) * self.scale
        return pts + torch.randn(n, 2, generator=generator) * self.noise

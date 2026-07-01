"""Wrap a distribution ``environment`` into a training source (the generic, env-agnostic dataset).

Materializes ``n_samples`` raw points once via ``environment.sample`` (the membrane normalizes them
later, in the runner). Exposes the :class:`BaseDataset` surface (``fit_tensor`` / ``num_items`` /
``sample_shape`` / ``gather``) so the runner can fit the preprocessor and pull batches. Works for any
environment that exposes ``sample(n, generator) -> (n, *sample_shape)`` — float32 (continuous) or
int64 (discrete token ids) — so all four sampling families reuse it.
"""

from __future__ import annotations

import torch

from forge.core.protocols import BaseDataset
from forge.core.registry import register
from forge.utils.seeding import make_generator


@register("dataset", "distribution")
class DistributionDataset(BaseDataset):
    def __init__(self, environment, n_samples: int = 10_000, seed: int = 0):
        gen = make_generator(seed)
        self.data: torch.Tensor = environment.sample(n_samples, generator=gen)
        self.dim = self.data.shape[-1]

    # ── BaseDataset surface (shared with TrajectoryDataset) ─────────────────────────────────────
    @property
    def fit_tensor(self) -> torch.Tensor:
        """Tensor the preprocessor fits on (per-feature stats)."""
        return self.data

    @property
    def num_items(self) -> int:
        return self.data.shape[0]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (self.dim,)

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        return self.data[idx.to(self.data.device)]  # idx may be on the train device; data lives on cpu

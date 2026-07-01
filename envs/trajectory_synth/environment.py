"""A dependency-free trajectory source: smooth wandering paths in a bounded 2-D box.

Each episode integrates a smoothly-varying (OU) velocity, producing locally smooth state paths the
diffuser can learn. Exposes ``rollouts()`` — a list of ``(L, dim)`` episodes — which the trajectory
dataset concatenates into a flat tensor and windows on the fly. A Minari/maze2d adapter can drop in
behind the `robotics` extra later, exposing the same ``rollouts()`` contract.
"""

from __future__ import annotations

from typing import List

import torch

from forge.core.registry import register
from forge.utils.seeding import make_generator


@register("environment", "trajectory_synth")
class SyntheticTrajectories:
    def __init__(
        self,
        dim: int = 2,
        episode_len: int = 200,
        n_episodes: int = 50,
        speed: float = 0.06,
        smooth: float = 0.85,
        bounds: float = 1.0,
        seed: int = 0,
    ):
        self.dim = int(dim)
        self.episode_len = int(episode_len)
        self.bounds = float(bounds)
        gen = make_generator(seed)

        self._rollouts: List[torch.Tensor] = []
        for _ in range(n_episodes):
            pos = (torch.rand(dim, generator=gen) * 2 - 1) * bounds
            vel = torch.randn(dim, generator=gen) * speed
            traj = [pos.clone()]
            for _ in range(self.episode_len - 1):
                vel = smooth * vel + (1 - smooth) * torch.randn(dim, generator=gen) * speed
                pos = (pos + vel).clamp(-bounds, bounds)
                traj.append(pos.clone())
            self._rollouts.append(torch.stack(traj))  # (L, dim)

    def rollouts(self) -> List[torch.Tensor]:
        return self._rollouts

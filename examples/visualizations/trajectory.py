"""Trajectory visualizer — scatter/line plot of plans in raw units (matplotlib, lazily imported).

Degrades gracefully: if matplotlib is absent it reports that instead of failing, so the core stays
dependency-free (matplotlib lives behind a `viz` extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

from forge.core.registry import register


@register("visualizer", "trajectory")
class TrajectoryVisualizer:
    def __init__(self, out_dir: str = "output"):
        self.out_dir = out_dir

    def render(self, plans: torch.Tensor, path: str = "plans.png") -> str | None:
        """Plot ``(N, H, dim)`` plans (first two dims). Returns the saved path, or None if no mpl."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return None

        plans = plans.detach().cpu()
        out = Path(self.out_dir) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        for tr in plans:
            ax.plot(tr[:, 0], tr[:, 1], alpha=0.6)
            ax.scatter([tr[0, 0]], [tr[0, 1]], c="green", s=20)
            ax.scatter([tr[-1, 0]], [tr[-1, 1]], c="red", s=20)
        ax.set_aspect("equal")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return str(out)

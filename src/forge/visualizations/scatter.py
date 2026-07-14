"""Scatter visualizer — 2-D scatter of generated samples in raw units (matplotlib, lazily imported).

Mirrors TrajectoryVisualizer: degrades gracefully when matplotlib is absent (returns None) so the
core stays dependency-free (matplotlib lives behind a `viz` extra). Plots the first two feature dims
of an ``(N, dim)`` sample batch; inputs that are not 2-D-plottable (e.g. token-id sequences) are
skipped, returning None — the runner hook is best-effort.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..core.registry import register


@register("visualizer", "scatter")
class ScatterVisualizer:
    def __init__(self, out_dir: str = "output"):
        self.out_dir = out_dir

    def render(self, samples: torch.Tensor, path: str = "samples.png") -> str | None:
        """Scatter ``(N, dim)`` samples (first two dims). Returns the saved path, or None when there
        is no matplotlib or the input is not 2-D-plottable."""
        samples = samples.detach().cpu()
        if samples.ndim != 2 or samples.shape[1] < 2:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return None

        out = Path(self.out_dir) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.4)
        ax.set_aspect("equal")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return str(out)

"""ModeCoverage — fraction of generated samples within `radius` of a cluster centroid.

Migrated verbatim from the runner's old inline fallback (TrainingRunner.evaluate). It needs the env
to expose `means` (only `gaussian_mixture` does). As an EXPLICITLY-configured metric it RAISES when
`means` is absent — a metric a user asked for must not silently vanish. (The runner keeps a soft
`hasattr(env, "means")` fallback for the *no-metric-configured* path; that is back-compat, not a
user metric.) Because of this hard-raise it is ENV-SPECIFIC — do not put it in a generic/shared
metric_set.
"""

from __future__ import annotations

import torch

from ..core.interfaces import Metric
from ..core.registry import register


@register("metric", "mode_coverage")
class ModeCoverage(Metric):
    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 radius: float = 0.6):
        super().__init__(environment, model, method, dataset, schedule)
        self.radius = float(radius)

    def __call__(self, samples=None, held_out=None) -> dict:
        if samples is None:
            raise ValueError("ModeCoverage is sample-driven but received samples=None.")
        if self.environment is None or not hasattr(self.environment, "means"):
            raise ValueError(
                "mode_coverage needs an environment exposing `means` (e.g. gaussian_mixture); "
                f"got {type(self.environment).__name__}. It is env-specific — don't put it in a "
                "generic metric_set."
            )
        x = samples.detach()
        means = self.environment.means.to(x.device)
        nearest = torch.cdist(x, means).min(dim=1).values
        return {"mode_coverage": float((nearest < self.radius).float().mean()),
                "radius": float(self.radius)}

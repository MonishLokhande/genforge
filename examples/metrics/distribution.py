"""Distribution-distance metrics: how close the generated samples are to the true distribution.

Sample-driven — each compares generated RAW-unit ``samples`` to a reference draw from the injected
``environment`` (``environment.sample(N)``), so they work on ANY env exposing ``sample`` (unlike
mode-coverage, which needs cluster ``means``). All pure-torch via ``torch.cdist`` except W2's exact
assignment, which needs scipy (the ``flow`` extra).
"""

from __future__ import annotations

import math

import torch

from forge.core.interfaces import Metric
from forge.core.registry import register


def _require_samples(samples, who: str) -> torch.Tensor:
    if samples is None:
        raise ValueError(f"{who} is a sample-driven metric but received samples=None.")
    return samples.detach()


def _reference(environment, n: int, device, seed: int) -> torch.Tensor:
    """A deterministic reference draw of true samples (fixed seed → stable across checkpoints)."""
    if environment is None or not hasattr(environment, "sample"):
        raise ValueError("distribution metric needs an `environment` exposing sample(n, generator).")
    gen = torch.Generator().manual_seed(int(seed))
    return environment.sample(n, generator=gen).to(device)


@register("metric", "mmd")
class MMD(Metric):
    """Gaussian-kernel MMD² between generated samples and an env reference draw.

    Bandwidth (comparability): default = median of pairwise distances over the REFERENCE draw only
    (the true distribution is fixed, so this is stable across checkpoints — a per-call median over
    the *pooled* set would drift as the model improves). Override with an explicit `bandwidths` list
    (a fixed multi-scale RBF sum). The bandwidth actually used is reported as `mmd_bandwidth`.
    """

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 n_ref: int = 0, ref_seed: int = 0, bandwidths=None):
        super().__init__(environment, model, method, dataset, schedule)
        self.n_ref = int(n_ref)
        self.ref_seed = int(ref_seed)
        self.bandwidths = list(bandwidths) if bandwidths else None

    def __call__(self, samples=None, held_out=None) -> dict:
        x = _require_samples(samples, "MMD")
        n = self.n_ref or x.shape[0]
        y = _reference(self.environment, n, x.device, self.ref_seed)
        dxx, dyy, dxy = torch.cdist(x, x), torch.cdist(y, y), torch.cdist(x, y)
        if self.bandwidths is None:
            iu = torch.triu_indices(y.shape[0], y.shape[0], offset=1)
            bws = [float(dyy[iu[0], iu[1]].median().clamp_min(1e-12))]  # reference-median heuristic
        else:
            bws = [float(b) for b in self.bandwidths]
        mmd2 = 0.0
        for bw in bws:
            g = 2.0 * bw * bw
            mmd2 += (torch.exp(-dxx**2 / g).mean() + torch.exp(-dyy**2 / g).mean()
                     - 2.0 * torch.exp(-dxy**2 / g).mean())
        mmd2 = float(mmd2 / len(bws))
        return {"mmd": mmd2, "mmd_bandwidth": float(sum(bws) / len(bws))}


@register("metric", "energy")
class EnergyDistance(Metric):
    """Energy distance E = 2·E|x−y| − E|x−x'| − E|y−y'| between samples and an env reference."""

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 n_ref: int = 0, ref_seed: int = 0):
        super().__init__(environment, model, method, dataset, schedule)
        self.n_ref = int(n_ref)
        self.ref_seed = int(ref_seed)

    def __call__(self, samples=None, held_out=None) -> dict:
        x = _require_samples(samples, "EnergyDistance")
        n = self.n_ref or x.shape[0]
        y = _reference(self.environment, n, x.device, self.ref_seed)
        e = 2.0 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean()
        return {"energy": float(e)}


@register("metric", "w2")
class Wasserstein2(Metric):
    """Exact empirical Wasserstein-2 via optimal assignment (scipy `linear_sum_assignment`).

    Caveats (recorded/documented): exact-assignment W2 between empirical samples is a biased,
    slowly-converging (dimension-sensitive) estimator, so values are comparable only at EQUAL N.
    O(n³) — both sides are capped to an equal `min(len, cap)`; the size used is reported as `w2_n`,
    which is the CAPPED size, NOT the dataset size (two runs both exceeding `cap` report the same
    `w2_n` even though their full-sample W2 could differ).
    """

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 cap: int = 2000, ref_seed: int = 0):
        super().__init__(environment, model, method, dataset, schedule)
        self.cap = int(cap)
        self.ref_seed = int(ref_seed)

    def __call__(self, samples=None, held_out=None) -> dict:
        try:
            from scipy.optimize import linear_sum_assignment
        except ModuleNotFoundError as e:  # match ot_cfm's lazy-scipy pattern
            raise ModuleNotFoundError("w2 metric needs SciPy (the flow extra): uv sync --extra flow.") from e
        x = _require_samples(samples, "Wasserstein2")
        y = _reference(self.environment, x.shape[0], x.device, self.ref_seed)
        n = min(x.shape[0], y.shape[0], self.cap)
        x, y = x[:n], y[:n]
        cost = (torch.cdist(x, y) ** 2).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        w2 = math.sqrt(float(cost[row, col].mean()))
        return {"w2": w2, "w2_n": float(n)}

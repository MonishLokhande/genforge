"""Flow-matching schedules — deterministic (probability-flow ODE, ``G = 0``) linear interpolants.

The reference process runs ``t: 0 → 1`` with the prior at ``t=0`` and data at ``t=1``. The marginal
is the same affine form every continuous schedule exposes, ``q(x_t | x_data) = N(α(t) x_data, σ(t)²)``,
so the **same** Euclidean forward primitive is reused — no `space` change is needed to add flow
(that's the Phase 2 proof that the paradigm axis is clean).

All output-type conversions — including the velocity conversions that need the path derivatives
`α̇, σ̇` — come from the `AffineGaussianContinuousSchedule` base; each flow supplies α/σ/α̇/σ̇ and
overrides only `G` (=0, probability-flow ODE) and the flow-direction grid (Invariant 3).
"""

from __future__ import annotations

import math

import torch

from ..core.interfaces import AffineGaussianContinuousSchedule
from ..core.registry import register
from ..utils.torch_utils import expand_like as _expand


class _AffineFlow(AffineGaussianContinuousSchedule):
    """Shared machinery for affine-path flows: x_t = α(t)·x_data + σ(t)·ε, velocity = α̇ x_data + σ̇ ε.

    Subclasses supply α/σ and their derivatives α̇/σ̇ (the base's abstract `alpha_dot`/`sigma_dot`);
    the base derives every output-type conversion, incl. velocity↔x0, from those four."""

    def G(self, t: torch.Tensor) -> torch.Tensor:
        # Probability-flow ODE: no diffusion. (Overrides the base's VP-derived G, which isn't
        # meaningful for a non-variance-preserving interpolant.)
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.zeros_like(t)

    def marginal(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = _expand(self.alpha(t), x0)
        s = _expand(self.sigma(t), x0)
        return a * x0, s.expand_as(x0)

    def discretize(self, n_steps: int) -> torch.Tensor:
        """Flow integrates 0 → 1 (prior → data)."""
        return torch.linspace(0.0, 1.0, n_steps + 1)


@register("schedule", "linear_flow")
class LinearFlow(_AffineFlow):
    """Rectified-flow linear interpolant: α(t)=t, σ(t)=1−t  ⟹  x_t = t·x_data + (1−t)·ε."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, dtype=torch.float32)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.as_tensor(t, dtype=torch.float32)

    def alpha_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.ones_like(t)

    def sigma_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return -torch.ones_like(t)


@register("schedule", "cfm_linear")
class CFMLinear(_AffineFlow):
    """Conditional-flow-matching path with a noise floor σ_min (Lipman et al.):
    α(t)=t, σ(t)=1−(1−σ_min)·t."""

    def __init__(self, sigma_min: float = 1e-2):
        self.sigma_min = float(sigma_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, dtype=torch.float32)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return 1.0 - (1.0 - self.sigma_min) * t

    def alpha_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.ones_like(t)

    def sigma_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return -(1.0 - self.sigma_min) * torch.ones_like(t)


@register("schedule", "si_trig")
class TrigInterpolant(_AffineFlow):
    """Variance-preserving trigonometric stochastic interpolant (Albergo & Vanden-Eijnden, 2023):
    α(t)=sin(πt/2), σ(t)=cos(πt/2), so α²+σ²=1 and the path runs from the Gaussian base (t=0,
    all-noise) to data (t=1). Affine, so it reuses the base output-type conversions; pairs with the
    `interpolant` SDE sampler, whose free diffusion coefficient ε(t) shares these same marginals."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.sin(0.5 * math.pi * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.cos(0.5 * math.pi * t)

    def alpha_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)

    def sigma_dot(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)

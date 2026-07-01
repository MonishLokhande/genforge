"""Variance-preserving (VP) linear and cosine schedules.

Continuous-time VP-SDE with:
  - Linear ``β(t) = β_min + t (β_max − β_min)`` (Song et al.) — ``VPLinear``
  - Cosine ᾱ(t) schedule (Nichol & Dhariwal, 2021) — ``VPCosine``

The marginal is ``q(x_t | x_0) = N(α(t) x_0, σ(t)² I)`` with the variance-preserving identity
``α(t)² + σ(t)² = 1``. DDPM is the time-discretization of this SDE.

The α/σ output-type conversion math lives in the ``ContinuousSchedule`` base (Invariant 3);
``VPLinear`` supplies β/α/σ/G, and ``VPCosine`` subclasses it, overriding only α/σ/G.
"""

from __future__ import annotations

import math

import torch

from ..core.interfaces import ContinuousSchedule
from ..core.registry import register
from ..utils.torch_utils import expand_like as _expand


@register("schedule", "vp_linear")
class VPLinear(ContinuousSchedule):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    # ── core schedule quantities ────────────────────────────────────────────────────────────────
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        # ∫_0^t β(s) ds = β_min t + ½ (β_max − β_min) t²
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.exp(-0.5 * self._integral_beta(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - a**2, min=1e-20))

    def G(self, t: torch.Tensor) -> torch.Tensor:
        """Forward-SDE diffusion coefficient G_t = sqrt(β(t))."""
        t = torch.as_tensor(t, dtype=torch.float32)
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = _expand(self.alpha(t), x0)
        s = _expand(self.sigma(t), x0)
        return a * x0, s.expand_as(x0)

    def discretize(self, n_steps: int) -> torch.Tensor:
        """Sampling time grid 1 → 0 (high noise → data). The reverse loop walks adjacent pairs."""
        return torch.linspace(1.0, 0.0, n_steps + 1)


@register("schedule", "vp_cosine")
class VPCosine(VPLinear):
    """Nichol–Dhariwal cosine VP schedule (Improved DDPM, 2021).

    Subclasses VPLinear to inherit marginal, discretize, σ, and all output-type conversions
    (pure α/σ math — Invariant 3). Overrides only α/G with the cosine curve (σ derives from α).

    ``parameterization`` selects the signal coefficient:
    - ``"alpha_bar"``: α(t) = f(t)/f(0) — the historical convention.
    - ``"sqrt_alpha_bar"``: α(t) = sqrt(f(t)/f(0)) — true Nichol–Dhariwal / diffusion-policy.
    Both are variance-preserving (α² + σ² = 1).
    """

    def __init__(self, s: float = 0.008, parameterization: str = "sqrt_alpha_bar"):
        # skip VPLinear.__init__ — beta_min/beta_max unused for cosine; set only what
        # the inherited conversions actually read (they only call self.alpha / self.sigma).
        if parameterization not in ("alpha_bar", "sqrt_alpha_bar"):
            raise ValueError(
                f"parameterization must be 'alpha_bar' or 'sqrt_alpha_bar', got {parameterization!r}"
            )
        self.s = float(s)
        self.parameterization = parameterization
        # f(0) depends only on s — precompute it once (alpha is called several times per step).
        self._f0 = self._f(torch.zeros((), dtype=torch.float32))

    def _f(self, t: torch.Tensor) -> torch.Tensor:
        """f(t) = cos²((t+s)/(1+s) · π/2) — the Nichol–Dhariwal base curve."""
        return torch.cos((t + self.s) / (1.0 + self.s) * math.pi / 2.0) ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        # ᾱ(t) = f(t)/f(0), clamped to [0, 1]
        a_bar = (self._f(t) / self._f0).clamp(max=1.0)
        return a_bar.sqrt() if self.parameterization == "sqrt_alpha_bar" else a_bar

    def G(self, t: torch.Tensor) -> torch.Tensor:
        """VP diffusion coefficient G(t) = sqrt(β(t)) where β(t) = -2 α'(t)/α(t).

        Derived via finite difference — closed-form is messy and not on the DDPM sampling path.
        finite-diff is sufficient; upgrade to analytical if numerical precision matters.
        """
        t = torch.as_tensor(t, dtype=torch.float32)
        dt = 1e-5
        t1 = (t + dt).clamp(max=1.0)
        t0 = (t - dt).clamp(min=0.0)
        dalpha_dt = (self.alpha(t1) - self.alpha(t0)) / (t1 - t0).clamp(min=dt)
        a = self.alpha(t).clamp(min=1e-10)
        beta = (-2.0 * dalpha_dt / a).clamp(min=0.0)
        return torch.sqrt(beta)

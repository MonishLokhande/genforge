"""Discrete (D3PM) schedules — a CTMC whose cumulative kernel Q̄_t carries data to a stationary
noise distribution, unifying the discrete cases.

Both absorbing (mask) and uniform diffusion share one structure: the marginal is
``Q̄_t[i,:] = ᾱ(t)·e_i + (1−ᾱ(t))·π`` with stationary π, and the one-step kernel from s→t has the
same convex form with β = ᾱ_t/ᾱ_s. So the reverse posterior is implemented once in the base and the
two schedules differ only in ``π`` and the (optional) mask index.

This is the discrete half of Invariant 1 — the continuous/discrete distinction lives ONLY here and
in `space`. Discrete prediction is always x₀ (logits) prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.interfaces import DiscreteSchedule
from ..core.registry import register


class _D3PM(DiscreteSchedule):
    """Shared D3PM machinery. Subclasses provide ``num_classes`` and the stationary ``π``."""

    num_classes: int

    def stationary(self) -> torch.Tensor:
        raise NotImplementedError

    # ᾱ(t): cumulative stay-probability, 1 at t=0 (data) → 0 at t=1 (noise).
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.as_tensor(t, dtype=torch.float32)

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noising hazard ``−ᾱ'(t)/(1−ᾱ(t))`` — the MDLM continuous-time NELBO weight.
        For the linear ᾱ(t)=1−t this is ``1/t`` (ᾱ'=−1, 1−ᾱ=t). Override if ᾱ changes."""
        t = torch.as_tensor(t, dtype=torch.float32)
        return 1.0 / t.clamp_min(1e-6)

    def alpha_bar_prime(self, t: torch.Tensor) -> torch.Tensor:
        """``d/dt ᾱ(t)``. Linear ᾱ=1−t ⟹ −1. (Override alongside `alpha_bar` if the schedule changes.)"""
        return -torch.ones_like(torch.as_tensor(t, dtype=torch.float32))

    def rate_matrix(self, t: torch.Tensor) -> torch.Tensor:
        """Forward CTMC generator ``Q_t = (ᾱ'/ᾱ)·(I − 1πᵀ)`` (rows sum to 0; off-diagonals are jump
        rates, diagonal is the negative total rate). One formula for both graphs via the stationary
        π — absorbing (π=e_mask) ⟹ rate into [MASK]; uniform (π=1/V) ⟹ uniform off-diagonal rate.

        Derived from the schedule's OWN ᾱ, so it is consistent with the forward marginals:
        ``Q_t @ Qbar(t) == dQbar/dt`` (Q_t = (dQbar/dt)·Qbar⁻¹, which collapses to this since 1πᵀ is
        idempotent). x₀-centric methods (D3PM/MDLM) ignore this; only rate-based SEDD reads it.

        Accepts scalar t → ``(V, V)`` or batched t ``(B,)`` → ``(B, V, V)``."""
        t = torch.as_tensor(t, dtype=torch.float32)
        coef = self.alpha_bar_prime(t) / self.alpha_bar(t).clamp_min(1e-6)     # ᾱ'/ᾱ  (negative)
        v = self.num_classes
        pi = self.stationary().to(coef.device)
        base = torch.eye(v, device=coef.device) - torch.ones(v, 1, device=coef.device) @ pi.unsqueeze(0)
        return coef.reshape(*coef.shape, 1, 1) * base                          # (..., V, V)

    def discretize(self, n_steps: int) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, n_steps + 1)

    def Qbar(self, t) -> torch.Tensor:
        ab = self.alpha_bar(t).reshape(())  # scalar t
        V = self.num_classes
        pi = self.stationary()
        return ab * torch.eye(V) + (1.0 - ab) * pi.unsqueeze(0).expand(V, V)

    def Q(self, t) -> torch.Tensor:
        # The stationary one-step kernel (rows = π); a valid stochastic matrix. The sampler uses
        # `reverse_probs`, not this directly.
        V = self.num_classes
        return self.stationary().unsqueeze(0).expand(V, V).clone()

    def _transition(self, s, t) -> torch.Tensor:
        """One-step kernel s→t: β·I + (1−β)·1πᵀ with β = ᾱ_t/ᾱ_s."""
        ab_s = self.alpha_bar(s).clamp_min(1e-8)
        ab_t = self.alpha_bar(t)
        beta = (ab_t / ab_s).reshape(())
        V = self.num_classes
        pi = self.stationary()
        return beta * torch.eye(V) + (1.0 - beta) * pi.unsqueeze(0).expand(V, V)

    def qt_probs(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(x_t = · | x_0) for each position: ``(..., V)``. Per-sample t supported."""
        V = self.num_classes
        onehot = F.one_hot(x0, V).to(torch.float32)               # (..., V), on x0's device
        ab = self.alpha_bar(t).to(onehot.device)
        ab = ab.reshape(list(ab.shape) + [1] * (onehot.ndim - ab.ndim))  # broadcast over (..., V)
        pi = self.stationary().to(onehot.device).reshape([1] * (onehot.ndim - 1) + [V])
        return ab * onehot + (1.0 - ab) * pi

    def reverse_probs(self, xt, t, s, x0_logits) -> torch.Tensor:
        """q(x_s | x_t) = Σ_{x0} q(x_s|x_t,x0) p_θ(x0|x_t), the factored D3PM posterior (..., V)."""
        V = self.num_classes
        p_x0 = torch.softmax(x0_logits, dim=-1)             # (..., V)
        qst = self._transition(s, t)                        # (V, V): [x_s, x_t]
        # fact1[..., k] = q(x_t | x_s = k) = qst[k, x_t]
        fact1 = qst.transpose(0, 1)[xt]                     # (..., V) over k
        qbar_s = self.Qbar(s)                               # (V, V)
        fact2 = p_x0 @ qbar_s                               # (..., V) = q(x_s | x0-dist)
        unnorm = fact1 * fact2
        return unnorm / unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@register("schedule", "uniform_discrete")
class UniformDiscrete(_D3PM):
    """Uniform diffusion: noise = uniform over all classes."""

    def __init__(self, num_classes: int = 4):
        self.num_classes = int(num_classes)

    def stationary(self) -> torch.Tensor:
        return torch.full((self.num_classes,), 1.0 / self.num_classes)


@register("schedule", "absorbing")
class AbsorbingDiffusion(_D3PM):
    """Absorbing (mask) diffusion: noise collapses all mass onto the mask token (the last index)."""

    def __init__(self, num_classes: int = 5, mask_index: int | None = None):
        self.num_classes = int(num_classes)
        self.mask_index = self.num_classes - 1 if mask_index is None else int(mask_index)

    def stationary(self) -> torch.Tensor:
        pi = torch.zeros(self.num_classes)
        pi[self.mask_index] = 1.0
        return pi

    def reverse_probs(self, xt, t, s, x0_logits) -> torch.Tensor:
        """Structured absorbing reverse — algebraically IDENTICAL to the base factored posterior, but
        without the dense ``Qbar``/``_transition`` (V,V) (10 GB at V=50258). Unmasked tokens are frozen;
        a masked token un-masks to data k ∝ (ᾱ_s−ᾱ_t)·p_θ(k) or stays masked ∝ (1−ᾱ_s)+ᾱ_s·p_θ(mask).
        All tensors are (…, V) (logits-scale)."""
        ab_t = float(self.alpha_bar(t))                            # scalars → device-agnostic
        ab_s = float(self.alpha_bar(s))
        p_x0 = torch.softmax(x0_logits, dim=-1)                     # (…, V), on the logits' device
        m = self.mask_index

        unnorm = (ab_s - ab_t) * p_x0                              # un-mask-to-data mass
        unnorm[..., m] = ab_s * p_x0[..., m] + (1.0 - ab_s)        # stay-masked mass
        probs_masked = unnorm / unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        delta = torch.zeros_like(p_x0).scatter(-1, xt.unsqueeze(-1), 1.0)   # frozen if already revealed
        return torch.where((xt == m).unsqueeze(-1), probs_masked, delta)

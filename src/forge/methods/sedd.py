"""SEDD (Lou et al.) — score-entropy discrete diffusion. CODE-FIRST: a different objective AND a
different reverse mechanism than x₀-prediction.

The model learns the **concrete score** s_θ(x_t)_y ≈ p_t(y)/p_t(x_t), trained with **denoising score
entropy** — the Bregman divergence ``ℓ(s, r) = s − r − r·log(s/r) ≥ 0`` (zero iff s = r) between s_θ
and the true conditional ratio ``r_y = q(y|x_0)/q(x_t|x_0)``.

**Why the naive form fails at LM vocab.** With ``s_θ = exp(raw)`` the loss carries an unnormalized
``Σ_y exp(score)`` over ~50k tokens — ill-conditioned and explosive (TinyStories: loss 203k, masked-
pred acc 0.0000 at V=50258; identical code learned fine at V=32). The paper's fix is a **bounded score
parameterization**: on the absorbing graph the un-masking score is a *scaled clean-token distribution*,
``s_θ(x_t)_y = (ᾱ/(1−ᾱ)) · softmax(logits)_y`` over data tokens (mask excluded, SUBS).

**The stable arrangement is a closed form (exploiting absorbing sparsity).** With that bounded score
and r a δ at x₀, the per-position score-entropy ``Σ_{y≠x_t} ℓ(s_y, r_y)`` collapses to ``−c·log p_θ(x₀)``
(``c = ᾱ/(1−ᾱ)``; the Σs and Σr terms cancel, only y=x₀ survives). The forward-rate weight β satisfies
``β·c = −ᾱ'/(1−ᾱ) = rate(t)``, so the objective is a **rate-weighted masked cross-entropy** — O(B·L),
no (B,L,V) score tensor, no Σ_y exp blow-up. The `score_entropy` integrand is kept as the definition
(Bregman-tested); the loss uses its closed form. FINDING (the special-case-collapses-to-general
principle): SEDD's bounded score on the absorbing graph reduces EXACTLY to MDLM's masked NELBO — the
two coincide here, differing only in the **reverse sampler** (SEDD's score-based analytic un-masking
vs. MDLM's tau-leaping x₀-posterior), which is what keeps SEDD a distinct paradigm.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.interfaces import Method, Model
from ..core.registry import register


def score_entropy(log_s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """The Bregman score-entropy integrand ``ℓ(s,r) = s − r − r·log(s/r)`` with ``s = exp(log_s)``.
    ≥0, =0 iff s=r. The loss does NOT sum this densely over the vocabulary (that is the ill-conditioned
    naive form); on the absorbing graph it uses ℓ's closed form (below). Kept as the objective's
    definition and guarded by the Bregman-divergence test."""
    s = log_s.clamp(max=20.0).exp()
    r_logr = r * r.clamp_min(1e-12).log()                 # r·log r  (→ 0 as r→0)
    return s - r - r * log_s + r_logr


@register("method", "sedd")
class SEDD(Method):
    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.t_eps = float(t_eps)
        self.mask_index = getattr(space, "mask_index", None)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,                                    # (B, L)
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b, length = x0.shape[0], x0.shape[1]
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - 2 * self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)

        logits = model(xt, t, cond)                          # (B, L, V) concrete-score logits
        if self.mask_index is not None:
            # In-place (no clone): final op is nn.Linear, which doesn't save its output for backward.
            logits[..., self.mask_index] = float("-inf")     # SUBS — never score the mask token
        log_p_x0 = torch.log_softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Bounded denoising score entropy on the absorbing graph, in CLOSED FORM (the paper's stable
        # arrangement + absorbing sparsity). The bounded score is s_θ = c·softmax(logits), c=ᾱ/(1−ᾱ),
        # and the conditional ratio r is a δ at x₀ with mass c. So at a masked position the score-entropy
        #   Σ_{y≠x_t} ℓ(s_y, r_y)  =  (Σ s) − (Σ r) − Σ r·log(s/r)  =  c − c − c·log p_θ(x₀)  =  −c·log p_θ(x₀)
        # (the Σs and Σr terms cancel; only y=x₀ survives the last sum). Its forward-rate weight β times
        # c is exactly the schedule hazard: β·c = (−ᾱ'/ᾱ)·(ᾱ/(1−ᾱ)) = −ᾱ'/(1−ᾱ) = rate(t). The whole
        # objective is therefore a rate-weighted cross-entropy on the un-masked tokens — O(B·L), no
        # (B,L,V) score tensor, no unnormalized Σ_y exp blow-up. (Revealed positions carry zero forward
        # rate, so they drop out.) FINDING: SEDD's bounded score on the absorbing graph reduces exactly
        # to MDLM's masked NELBO; the two paradigms coincide here, and differ only in the reverse sampler.
        masked = (xt == self.mask_index).to(log_p_x0.dtype) if self.mask_index is not None \
            else torch.ones_like(log_p_x0)
        weight = self.schedule.rate(t)                       # (B,) = β·c
        return (weight * (-log_p_x0 * masked).sum(dim=1)).mean() / length

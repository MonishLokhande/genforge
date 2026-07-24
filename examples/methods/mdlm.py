"""MDLM (Sahoo et al.) — masked diffusion LM objective.

The absorbing-diffusion continuous-time NELBO reduces to a cross-entropy over the MASKED positions
only, weighted by the noising hazard ``rate(t) = −ᾱ'(t)/(1−ᾱ(t))``, with the SUBS parameterization
(the model never predicts the mask token). That masked-only restriction + the rate weight are the
ONLY difference from `d3pm`'s plain all-position x₀-CE; the absorbing schedule/space and the
tau_leaping reverse are reused. Faithful to the paper's objective, not its repo.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from forge.core.interfaces import Method, Model
from forge.core.registry import register


@register("method", "mdlm")
class MDLM(Method):
    nelbo = True  # loss is the masked continuous-time NELBO → convertible to bits/perplexity
    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.t_eps = float(t_eps)
        self.mask_index = getattr(space, "mask_index", None)
        if self.mask_index is None:
            raise ValueError("MDLM requires an absorbing/masking space (`space.mask_index` set).")

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,                                    # (B, L) clean tokens
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b, length = x0.shape[0], x0.shape[1]
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - 2 * self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)

        # In-place mask-out (no clone): the model's final op is nn.Linear, whose backward saves its
        # input/weight, not this output — so editing it is exact (verified bit-identical gradients).
        logits = model(xt, t, cond)
        logits[..., self.mask_index] = float("-inf")         # SUBS: never predict the mask token
        v = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, v), x0.reshape(-1), reduction="none").reshape(b, length)

        masked = (xt == self.mask_index).to(ce.dtype)        # loss only on corrupted positions
        weight = self.schedule.rate(t)                       # (B,) continuous-time NELBO weight
        return (weight * (ce * masked).sum(dim=1)).mean() / length

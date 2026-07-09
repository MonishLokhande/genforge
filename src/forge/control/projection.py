"""Projection-family controllers — hard constraints expressed as control (surface="x0").

`Projection` projects the clean estimate x̂₀ onto a cost's feasible set each reverse step. `Pin` is the
degenerate case: a hard EQUALITY constraint pinning known entries to observed values — which is exactly
what inpainting / goal-conditioning does at sampling time, so it lives in the control layer rather than
as bespoke sampler logic. The base model is never touched (Invariant 6).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.interfaces import Controller
from ..core.registry import register


@register("control", "projection")
class Projection(Controller):
    surface = "x0"

    def modify_x0(self, x0_hat: torch.Tensor, x: torch.Tensor, t, schedule, cond=None, context=None) -> torch.Tensor:
        return self.cost.project(x0_hat)


@register("control", "pin")
class Pin(Controller):
    """Pin known entries to observed values — a hard EQUALITY constraint = a projection (surface="x0").

    Inpainting / goal-conditioning at sampling time IS this: pinning start/goal (or feature columns)
    every reverse step. So it routes through the control layer, in the sampler's dedicated LAST (pin)
    slot — applied AFTER the primary controller so pins still WIN.

    Byte-parity (surfaced nuance): a pure x̂₀ pin is re-noised by the reverse step at every step except
    the final noiseless one, so it can't keep endpoints byte-exact mid-trajectory. The pin therefore
    lands on the SAMPLE in the pin slot (`project(x)`, exact every step) — which is why the two-slot
    structure (primary control, then pin) is kept rather than folded into a single x̂₀ projection.
    """

    surface = "x0"

    def __init__(self, cost=None, mask=None, values=None, idx=None, val=None):
        super().__init__(cost)
        self._mask, self._values, self._idx, self._val = mask, values, idx, val

    @classmethod
    def from_cond(cls, cond) -> "Optional[Pin]":
        """Build the pin from a sampler conditioning spec, or None if there is nothing to pin."""
        if isinstance(cond, dict):
            if "inpaint" in cond:
                mask, values = cond["inpaint"]
                return cls(mask=mask, values=values)
            if "pin" in cond:
                idx, val = cond["pin"]
                return cls(idx=idx, val=val)
        return None

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Set the pinned entries of `x` to their observed values (the hard equality projection)."""
        if self._mask is not None:
            return torch.where(self._mask.to(x.device), self._values.to(x.device, x.dtype), x)
        x = x.clone()
        x[..., self._idx] = self._val
        return x

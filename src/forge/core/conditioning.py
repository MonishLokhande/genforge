"""Boundary-pinned conditioning — the `Pin` controller the base sample loop relies on.

`Pin` lives in ``core`` (not among the swappable control examples) precisely because the framework's
own reverse loop uses it: ``Sampler._apply_conditioning`` pins conditioned entries to observed
values every reverse step (inpainting / goal-conditioning IS a hard equality constraint = control,
Invariant 6). Keeping it here lets the core stand alone — the loop never imports a concrete example.
"""

from __future__ import annotations

from typing import Optional

import torch

from .interfaces import Controller
from .registry import register


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

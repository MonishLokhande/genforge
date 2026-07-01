"""Exponential moving average of model weights.

The EMA shadow is embedded in every checkpoint and used automatically at sample time.
Kept deliberately small and explicit.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """Tracks an EMA shadow of a module's parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 10):
        self.decay = decay
        # Warmup makes the effective decay ramp from ~0 to `decay`, so the shadow tracks the model
        # early and converges fast on short runs (decay matched to run length).
        self.warmup = warmup
        self.num_updates = 0
        # Shadow lives on the same device as the model; detached clones.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup: dict[str, torch.Tensor] | None = None

    def _effective_decay(self) -> float:
        if self.warmup <= 0:
            return self.decay
        return min(self.decay, (self.num_updates + 1) / (self.num_updates + 1 + self.warmup))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._effective_decay()
        self.num_updates += 1
        # Collect the float tensors, then fuse the EMA (s = d·s + (1−d)·v) into two _foreach_ kernels
        # instead of ~2N tiny per-tensor launches. Bit-identical to s.mul_(d).add_(v, alpha=1−d).
        # assumes the float shadow is single-dtype/device (true here; all fp32, one device) —
        # _foreach_ would otherwise need per-(dtype,device) grouping.
        shadow_floats, model_floats = [], []
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow_floats.append(s)
                model_floats.append(v)
            else:
                s.copy_(v)               # integer buffers (e.g. counters) are copied, not averaged
        if shadow_floats:
            torch._foreach_mul_(shadow_floats, d)
            torch._foreach_add_(shadow_floats, model_floats, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model`` weights with the EMA shadow (in place)."""
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Stash current model weights so they can be restored after sampling."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA.restore called without a prior EMA.store.")
        model.load_state_dict(self._backup, strict=True)
        self._backup = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": deepcopy(self.shadow),
        }

    def load_state_dict(self, d: dict) -> None:
        self.decay = d["decay"]
        self.warmup = d.get("warmup", 0)
        self.num_updates = d.get("num_updates", 0)
        self.shadow = {k: v.clone() for k, v in d["shadow"].items()}

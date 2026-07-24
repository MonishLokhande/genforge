"""Amortized value control — guide sampling by ∇V of a LEARNED value model.

The value model is loaded LAZILY from a checkpoint written by a paired training `method`
(value_training / fbsde). The checkpoint is the ONLY channel between method and controller
(Invariant 6): this module imports the registry + checkpoint loader, never a method. The value
model is reconstructed from the checkpoint's embedded config and frozen.
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core import registry
from forge.core.checkpoint import load_checkpoint
from forge.core.interfaces import Controller
from forge.core.registry import register


class AmortizedValueController(Controller):
    """Loads a value model from a checkpoint and bends x̂₀ along ``sign·∇V`` (σ(t)²-faded).

    An x0-surface controller (a clean-estimate shift). The artifact is loaded in `prepare` — the
    runner calls it before sampling — with a lazy `_load()` safety net for direct (no-runner) use.
    The checkpoint is the ONLY channel to the paired training method (Invariant 6)."""

    surface = "x0"

    def __init__(
        self,
        value_checkpoint: str,
        cost=None,
        scale: float = 1.0,
        sigma_weight: bool = True,
        sign: float = 1.0,
    ):
        super().__init__(cost)
        self.value_checkpoint = value_checkpoint
        self.scale = float(scale)
        self.sigma_weight = bool(sigma_weight)
        self.sign = float(sign)
        self._value: Optional[torch.nn.Module] = None

    def prepare(self, preprocessor) -> None:
        super().prepare(preprocessor)
        self._load()                     # amortized controllers load their artifact here

    def _load(self) -> None:
        if self._value is not None:
            return
        ckpt = load_checkpoint(self.value_checkpoint)
        cfg = ckpt.get("config") or {}
        mcfg = cfg["model"]
        model = registry.create("model", mcfg["name"], **(mcfg.get("params") or {}))
        ema = ckpt.get("ema_state")
        state = ema["shadow"] if ema is not None else ckpt["model_state"]
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._value = model

    def modify_x0(self, x0_hat: torch.Tensor, x: torch.Tensor, t, schedule, cond=None, context=None) -> torch.Tensor:
        self._load()
        # The Model contract is forward(x, t, cond=None) and value nets are noise-level aware, so t
        # is REQUIRED — value_mlp happens to accept and ignore it, but e.g. value_unet does not.
        # Samplers pass t as a scalar or a per-sample tensor; broadcast to the batch either way.
        tt = torch.as_tensor(t, device=x0_hat.device, dtype=torch.float32).reshape(-1)
        if tt.numel() == 1:
            tt = tt.expand(x0_hat.shape[0])
        with torch.enable_grad():
            z = x0_hat.detach().requires_grad_(True)
            v = self._value(z, tt).sum()
            (grad,) = torch.autograd.grad(v, z)
        step = self.scale * self.sign
        if self.sigma_weight:
            # σ(t) is per-sample when t is; reshape to (B, 1, …) so it broadcasts against x̂₀ of any
            # rank — a bare (B,) sigma collides with the feature dim on trajectories (B, H, D).
            sigma = torch.as_tensor(schedule.sigma(tt), device=x0_hat.device)
            step = step * sigma.reshape(-1, *([1] * (x0_hat.dim() - 1))) ** 2
        return x0_hat + step * grad


@register("control", "value_guidance")
class ValueGuidance(AmortizedValueController):
    """Bias sampling TOWARD high learned value (``sign = +1``)."""

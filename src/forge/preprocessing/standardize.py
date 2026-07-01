"""Per-feature standardization: x̃ = (x − μ) / σ.

An **affine** membrane — the Jacobian is constant (diag(1/σ)) — so a real-unit cost maps exactly
into normalized space (Invariant 8). Touches only the generated quantity (Invariant 9). Stats
travel in the checkpoint (Invariant 5).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Preprocessor
from ..core.registry import register


@register("preprocessor", "standardize")
class Standardize(Preprocessor):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> None:
        data = data.reshape(-1, data.shape[-1])
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0).clamp_min(self.eps)

    def _check(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardize used before fit()/load_state_dict().")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self._check()
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, x_tilde: torch.Tensor) -> torch.Tensor:
        self._check()
        return x_tilde * self.std.to(x_tilde.device) + self.mean.to(x_tilde.device)

    # ── affine map exposed for cost.to_normalized (Invariant 8) ─────────────────────────────────
    # x̃ = scale ⊙ x + shift  with  scale = 1/σ,  shift = −μ/σ.
    def scale(self) -> torch.Tensor:
        self._check()
        return 1.0 / self.std

    def shift(self) -> torch.Tensor:
        self._check()
        return -self.mean / self.std

    def state_dict(self) -> dict:
        self._check()
        return {"mean": self.mean.cpu().clone(), "std": self.std.cpu().clone(), "eps": self.eps}

    def load_state_dict(self, d: dict) -> None:
        self.mean = torch.as_tensor(d["mean"])
        self.std = torch.as_tensor(d["std"])
        self.eps = float(d.get("eps", 1e-6))

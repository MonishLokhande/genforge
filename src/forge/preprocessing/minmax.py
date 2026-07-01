"""Per-feature min-max scaling to ``[-1, 1]`` (the diffuser convention).

Affine like `Standardize` (constant Jacobian), so real-unit costs push through exactly
(Invariant 8). Touches only the generated quantity (Invariant 9).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Preprocessor
from ..core.registry import register


@register("preprocessor", "minmax")
class MinMax(Preprocessor):
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> None:
        data = data.reshape(-1, data.shape[-1])
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values

    def _range(self) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise RuntimeError("MinMax used before fit()/load_state_dict().")
        return (self.max - self.min).clamp_min(self.eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        rng = self._range().to(x.device)
        return 2.0 * (x - self.min.to(x.device)) / rng - 1.0

    def inverse(self, x_tilde: torch.Tensor) -> torch.Tensor:
        rng = self._range().to(x_tilde.device)
        return (x_tilde + 1.0) / 2.0 * rng + self.min.to(x_tilde.device)

    # x̃ = scale ⊙ x + shift  with  scale = 2/range,  shift = −1 − 2·min/range.
    def scale(self) -> torch.Tensor:
        return 2.0 / self._range()

    def shift(self) -> torch.Tensor:
        return -1.0 - 2.0 * self.min / self._range()

    def state_dict(self) -> dict:
        self._range()  # validates fitted
        return {"min": self.min.cpu().clone(), "max": self.max.cpu().clone(), "eps": self.eps}

    def load_state_dict(self, d: dict) -> None:
        self.min = torch.as_tensor(d["min"])
        self.max = torch.as_tensor(d["max"])
        self.eps = float(d.get("eps", 1e-6))

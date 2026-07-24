"""A small MLP field for low-dimensional targets, with sinusoidal time conditioning.

`output_type` is configurable (default ``eps``). The model is otherwise paradigm-agnostic — what
its output *means* is interpreted by the schedule's conversions, not here (Invariant 3).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..core.interfaces import Model
from ..core.registry import register
from ..nn import sinusoidal_embedding


@register("model", "mlp")
class MLP(Model):
    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        depth: int = 3,
        output_type: str = "eps",
        time_embed_dim: int = 32,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.output_type = output_type
        self.time_embed_dim = time_embed_dim
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        in_dim = dim + hidden + cond_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_cond(cond)
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        parts = [x, temb]
        if self.cond_dim and cond is not None:
            parts.append(cond)
        return self.net(torch.cat(parts, dim=-1))

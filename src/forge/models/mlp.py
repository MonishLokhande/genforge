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


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding of a scalar time in [0, 1]."""
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero-pad to an odd dim
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


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
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))
        parts = [x, temb]
        if self.cond_dim and cond is not None:
            parts.append(cond)
        return self.net(torch.cat(parts, dim=-1))

"""A small logits model for discrete tokens: embed + time-condition + per-position MLP head.

`output_type = "logits"`. Position-independent (sufficient for the toy categorical target); a
sequence model (transformer) can drop in later behind the same contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from forge.core.interfaces import Model
from forge.core.registry import register
from forge.nn import sinusoidal_embedding


@register("model", "categorical_mlp")
class CategoricalMLP(Model):
    output_type = "logits"

    def __init__(
        self,
        num_classes: int = 5,
        hidden: int = 128,
        depth: int = 3,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_embed_dim = time_embed_dim

        self.embed = nn.Embedding(num_classes, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        layers: list[nn.Module] = []
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_cond(cond)
        # x: (B, L) long tokens
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        h = self.embed(x)                                            # (B, L, hidden)
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))  # (B, hidden)
        h = h + temb.unsqueeze(1)
        return self.head(h)                                          # (B, L, num_classes)

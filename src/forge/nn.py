"""Small neural-net building blocks shared across models — framework infra, not a model.

These are helpers that more than one architecture needs (so they can't live inside any single
reference model). Kept deliberately tiny; a model is still ~30-60 readable lines on top of these.
"""

from __future__ import annotations

import math

import torch


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

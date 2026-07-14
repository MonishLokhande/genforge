"""Small torch helpers shared across components."""

from __future__ import annotations

import torch
import torch.nn as nn


def model_device(model: nn.Module) -> torch.device:
    """Device a model lives on; falls back to CPU for parameter-less models."""
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def expand_like(coeff, ref: torch.Tensor) -> torch.Tensor:
    """View a per-batch (or scalar) coefficient so it broadcasts over ``ref``'s trailing dims."""
    coeff = torch.as_tensor(coeff, device=ref.device)
    while coeff.ndim < ref.ndim:
        coeff = coeff.unsqueeze(-1)
    return coeff


def cond_to(cond, device):
    """Move a conditioning spec's tensors to ``device``. Tensor | dict | None.

    A bare ``if torch.is_tensor(cond)`` silently leaves a DICT cond on the CPU, which strands
    image conditioning away from a CUDA model. Non-tensor dict values (e.g. `Pin`'s
    ``{"pin": (indices, values)}`` tuple) pass through untouched.
    """
    if torch.is_tensor(cond):
        return cond.to(device)
    if isinstance(cond, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}
    return cond

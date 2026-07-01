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

"""Flat-tensor trajectory windowing (≈90× memory saving vs. a materialized window tensor).

All episodes are concatenated once into a flat ``(ΣL, dim)`` device tensor. Windows are gathered
**on the fly** from per-window start indices (plus episode bounds for padding) — the materialized
``(num_windows, H, dim)`` tensor is never built, so memory stays ≈ ``ΣL·dim`` rather than
``num_windows·H·dim``. Normalize-once happens in the runner via the preprocessor fitted on
``fit_tensor`` (the flat tensor).
"""

from __future__ import annotations

import torch

from forge.core.protocols import BaseDataset
from forge.core.registry import register


@register("dataset", "trajectory")
class TrajectoryDataset(BaseDataset):
    def __init__(
        self,
        environment,
        horizon: int = 32,
        use_padding: bool = False,
        pad_before: int = 0,
        pad_after: int = 0,
    ):
        rollouts = environment.rollouts()
        self.dim = int(rollouts[0].shape[-1])
        self.horizon = int(horizon)
        self.flat = torch.cat(rollouts, dim=0).to(torch.float32)  # (ΣL, dim)

        starts, ep_lo, ep_hi = [], [], []
        offset = 0
        for r in rollouts:
            L = r.shape[0]
            if use_padding:
                lo, hi = -pad_before, L - self.horizon + pad_after
            else:
                lo, hi = 0, L - self.horizon
            for s in range(lo, hi + 1):
                starts.append(offset + s)
                ep_lo.append(offset)
                ep_hi.append(offset + L)  # exclusive
            offset += L
        # Only three small (num_windows,) index tensors are stored — never the windows themselves.
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.ep_lo = torch.tensor(ep_lo, dtype=torch.long)
        self.ep_hi = torch.tensor(ep_hi, dtype=torch.long)

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Materialize a batch of windows ``(B, H, dim)`` on the fly from the flat tensor."""
        idx = idx.to(self.starts.device)
        s = self.starts[idx]                                   # (B,)
        lo = self.ep_lo[idx].unsqueeze(1)
        hi = self.ep_hi[idx].unsqueeze(1)
        offs = torch.arange(self.horizon, device=s.device).unsqueeze(0)  # (1, H)
        gidx = (s.unsqueeze(1) + offs).clamp(lo, hi - 1)       # (B, H), pad by repeating boundary
        return self.flat.to(gidx.device)[gidx]                # (B, H, dim)

    # ── BaseDataset surface ─────────────────────────────────────────────────────────────────────
    @property
    def fit_tensor(self) -> torch.Tensor:
        return self.flat

    @property
    def num_items(self) -> int:
        return self.starts.shape[0]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (self.horizon, self.dim)

    def stored_bytes(self) -> int:
        """Footprint of the preloaded representation (flat tensor + index tensors)."""
        return (
            self.flat.numel() * self.flat.element_size()
            + sum(t.numel() * t.element_size() for t in (self.starts, self.ep_lo, self.ep_hi))
        )

    def materialized_bytes(self) -> int:
        """Footprint the naive (num_windows, H, dim) materialization WOULD take."""
        return self.num_items * self.horizon * self.dim * self.flat.element_size()

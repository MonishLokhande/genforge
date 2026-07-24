"""A temporal (1-D conv) U-Net over the time axis — the diffuser backbone for trajectory windows.

Operates on ``(B, H, x_dim)`` windows (``x_dim`` may be ``[action | state]``), internally treating
time ``H`` as the conv length and ``x_dim`` as channels. `output_type` configurable (default eps).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from forge.core.interfaces import Model
from forge.core.registry import register
from forge.nn import sinusoidal_embedding


def _groupnorm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class _ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(_groupnorm(in_ch), nn.SiLU(), nn.Conv1d(in_ch, out_ch, 3, padding=1))
        self.time = nn.Linear(time_dim, out_ch)
        self.block2 = nn.Sequential(_groupnorm(out_ch), nn.SiLU(), nn.Conv1d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time(temb).unsqueeze(-1)          # FiLM-style additive time conditioning
        h = self.block2(h)
        return h + self.skip(x)


@register("model", "temporal_unet")
class TemporalUNet(Model):
    def __init__(
        self,
        dim: int = 2,
        horizon: int = 32,
        base: int = 32,
        dim_mults=(1, 2, 4),
        output_type: str = "eps",
        time_embed_dim: int = 32,
    ):
        super().__init__()
        factor = 2 ** len(dim_mults)
        if horizon % factor != 0:
            raise ValueError(
                f"horizon={horizon} must be divisible by 2**len(dim_mults)={factor} so the U-Net's "
                f"down/up samples line up (got remainder {horizon % factor})."
            )
        self.dim = dim
        self.horizon = horizon
        self.output_type = output_type
        self.time_embed_dim = time_embed_dim

        time_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.in_proj = nn.Conv1d(dim, base, 3, padding=1)

        chans = [base * m for m in dim_mults]
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base
        for ch in chans:
            self.downs.append(_ResBlock1d(prev, ch, time_dim))
            self.downsamples.append(nn.Conv1d(ch, ch, 4, stride=2, padding=1))
            prev = ch

        self.mid = _ResBlock1d(prev, prev, time_dim)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for ch in reversed(chans):
            self.upsamples.append(nn.ConvTranspose1d(prev, ch, 4, stride=2, padding=1))
            self.ups.append(_ResBlock1d(ch * 2, ch, time_dim))  # concat skip
            prev = ch

        self.out = nn.Sequential(_groupnorm(prev), nn.SiLU(), nn.Conv1d(prev, dim, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_cond(cond)
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim))

        h = self.in_proj(x.transpose(1, 2))                  # (B, base, H)
        skips = []
        for block, down in zip(self.downs, self.downsamples):
            h = block(h, temb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, temb)
        for up, block in zip(self.upsamples, self.ups):
            h = up(h)
            h = block(torch.cat([h, skips.pop()], dim=1), temb)
        out = self.out(h)
        return out.transpose(1, 2)                           # (B, H, dim)

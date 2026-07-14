"""Value function (temporal encoder + scalar head) — forge port of the reference implementation's
``value_function``, checkpoint-compatible with the trained locomotion value heads.

Janner ValueFunction layout: per-stage ``(ResBlock, ResBlock, Downsample)`` — EVERY stage
downsamples (unlike the U-Net, whose last stage doesn't) — then two mid blocks halving
channels with downsamples, then ``Linear(fc_dim + time_dim) -> Mish -> Linear -> out``.
Block internals reuse forge's Janner module (proven key-identical to the reference implementation).
The head is sized from a real dummy forward (the reference implementation's documented fix for non-power-of-2
horizons). forward conforms to the forge Model contract ``(x, t, cond=None)``; the reference implementation's
original signature was ``(x, cond, t)`` with cond unused.

# ⚠ CHECKPOINT PARITY — module/parameter names frozen; do not rename.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.core.interfaces import Model
from forge.core.registry import register
from forge.models.temporal_unet_janner import (
    Downsample1d,
    ResidualTemporalBlock,
    _SinusoidalPosEmb,
)


@register("model", "value_unet")
class ValueUNet(Model):
    """Temporal encoder with a scalar value head. ``forward(x, t) -> (B, out_dim)``."""

    output_type: str

    def __init__(
        self,
        *,
        horizon: int,
        transition_dim: int,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        out_dim: int = 1,
        output_type: str = "value",
    ):
        super().__init__()
        self.output_type = output_type
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.dim = int(dim)
        self.dim_mults = tuple(int(m) for m in dim_mults)
        self.out_dim = int(out_dim)

        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = self.dim

        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(self.dim),
            nn.Linear(self.dim, self.dim * 4),
            nn.Mish(),
            nn.Linear(self.dim * 4, self.dim),
        )

        self.blocks = nn.ModuleList([])
        for dim_in, dim_out in in_out:
            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Downsample1d(dim_out),
            ]))

        mid_dim = dims[-1]
        mid_dim_2, mid_dim_3 = mid_dim // 2, mid_dim // 4
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, embed_dim=time_dim)
        self.mid_down1 = Downsample1d(mid_dim_2)
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, embed_dim=time_dim)
        self.mid_down2 = Downsample1d(mid_dim_3)

        fc_dim = self._infer_fc_dim()
        hidden = max(fc_dim // 2, 1)
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, self.out_dim),
        )

    def _encode(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for resnet, resnet2, downsample in self.blocks:
            h = resnet(h, t_emb)
            h = resnet2(h, t_emb)
            h = downsample(h)
        h = self.mid_block1(h, t_emb)
        h = self.mid_down1(h)
        h = self.mid_block2(h, t_emb)
        h = self.mid_down2(h)
        return h

    def _infer_fc_dim(self) -> int:
        with torch.no_grad():
            h = self._encode(torch.zeros(1, self.transition_dim, self.horizon),
                             torch.zeros(1, self.dim))
        return int(h.shape[1] * h.shape[2])

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        if x.shape[1] != self.horizon or x.shape[2] != self.transition_dim:
            raise ValueError(
                f"expected x of shape (B,{self.horizon},{self.transition_dim}), got {tuple(x.shape)}"
            )
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t_emb = self.time_mlp(t)
        h = x.permute(0, 2, 1).contiguous()
        h = self._encode(h, t_emb)
        h = h.reshape(h.shape[0], -1)
        return self.final_block(torch.cat([h, t_emb], dim=-1))

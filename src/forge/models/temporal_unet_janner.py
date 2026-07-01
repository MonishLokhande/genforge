"""Janner TemporalUNet — faithful port of the Diffuser trajectory denoiser.

Module structure and parameter names are frozen to match the reference Janner implementation
so trained checkpoints load with strict=True on the U-Net body. Do NOT rename any nn.Module
subclass or parameter without verifying against a real checkpoint's state_dict keys.

Reference: Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" (Diffuser).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..core.interfaces import Model
from ..core.registry import register


# ── parameter-free shape helpers (indices must stay stable for Conv1dBlock.block) ──

class _ExpandDim(nn.Module):
    """(B,C,L) -> (B,C,1,L) — sits at block index 1, no params."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-2)


class _SqueezeLastDim(nn.Module):
    """(B,C,1,L) -> (B,C,L) — sits at block index 3, no params."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-2)


class _AppendDim(nn.Module):
    """(B,C) -> (B,C,1) — sits at time_mlp index 2, no params."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


# ── sinusoidal embedding (non-persistent buffer → NOT in state_dict) ──────────

class _SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding (B,) -> (B, dim)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        scale = math.log(10000) / (half - 1)
        # non-persistent so it doesn't appear in state_dict (matches Janner ckpt)
        self.register_buffer("freqs", torch.exp(torch.arange(half) * -scale), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = x[:, None].float() * self.freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ── Conv1dBlock: Conv1d(idx=0) → ExpandDim(1) → GroupNorm(2) → Squeeze(3) → Mish(4) ──

class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish over the temporal axis.

    Sub-indices within `self.block` are frozen to checkpoint parity:
      0: Conv1d  (has weights/bias)
      1: ExpandDim  (no params)
      2: GroupNorm  (has weight/bias)
      3: SqueezeLastDim  (no params)
      4: Mish  (no params)
    """
    def __init__(self, inp: int, out: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            _ExpandDim(),
            nn.GroupNorm(n_groups, out),
            _SqueezeLastDim(),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── up/down samplers ──────────────────────────────────────────────────────────

class Downsample1d(nn.Module):
    """Stride-2 1D conv downsampler — key: `.conv.*`."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Stride-2 transposed-conv upsampler — key: `.conv.*`."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── ResidualTemporalBlock ─────────────────────────────────────────────────────

class ResidualTemporalBlock(nn.Module):
    """Two Conv1dBlocks with time(/cond) FiLM injection and a 1×1 residual conv.

    time_mlp sub-indices:
      0: Mish  (no params)
      1: Linear(embed_dim, emb_out)  (has weight/bias → `time_mlp.1.*`)
      2: AppendDim  (no params)

    cond_predict_scale=False (default, frozen Janner path): additive (bias-only FiLM).
    cond_predict_scale=True: scale+bias FiLM — emb_out = 2*out_channels.
    """
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        emb_out = out_channels * 2 if self.cond_predict_scale else out_channels
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, emb_out),
            _AppendDim(),
        )
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, kernel_size=1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t)
        out = self.blocks[0](x)
        if self.cond_predict_scale:
            emb = emb.reshape(emb.shape[0], 2, self.out_channels, 1)
            out = emb[:, 0] * out + emb[:, 1]
        else:
            out = out + emb
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ── ObsNormalizer (Invariant 9: model owns obs normalization, not the preprocessor) ──

class ObsNormalizer(nn.Module):
    """Per-dimension standardizer for observation conditioning.

    Buffers are PERSISTENT so they travel in state_dict (Invariant 5). The model calls
    this on `cond`'s obs slice before FiLM — never the membrane preprocessor (Invariant 9).
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))

    def fit(self, obs: torch.Tensor) -> None:
        """Compute mean/std from obs (B, obs_dim) and store as persistent buffers."""
        self.mean.copy_(obs.mean(0))
        self.std.copy_(obs.std(0, unbiased=False).clamp(min=1e-6))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std


# ── TemporalUNetJanner ────────────────────────────────────────────────────────

@register("model", "temporal_unet_janner")
class TemporalUNetJanner(Model):
    """Janner TemporalUNet — faithful port for checkpoint-compatible trajectory denoising.

    U-Net body state_dict keys and shapes match the reference trained checkpoints exactly,
    enabling strict=True body loading from those checkpoints.

    Args:
        horizon:           trajectory length H.
        transition_dim:    channel width D (action_dim + obs_dim in the Janner layout).
        dim:               base channel count (e.g. 32).
        dim_mults:         channel multipliers per resolution — REQUIRED, no default (maze2d uses
                           (1,4,8), locomotion uses (1,2,4,8); wrong default silently breaks ckpt loads).
        cond_dim:          global conditioning vector size; 0 = unconditional.
        cond_predict_scale: FiLM scale+bias (True) vs bias-only (False, Janner default).
        output_type:       genforge Model contract ("x0", "eps", "score", …).
        obs_dim:           if >0, a persistent ObsNormalizer is constructed for the obs
                           slice of cond; its keys are obs_normalizer.* and are NOT
                           present in the reference checkpoints (expected extras).
    """

    output_type: str

    def __init__(
        self,
        *,
        horizon: int,
        transition_dim: int,
        dim: int = 32,
        dim_mults: tuple[int, ...],
        cond_dim: int = 0,
        cond_predict_scale: bool = False,
        output_type: str = "x0",
        obs_dim: int = 0,
    ):
        super().__init__()
        self.output_type = output_type
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.dim = int(dim)
        self.dim_mults = tuple(int(m) for m in dim_mults)
        self.cond_dim = int(cond_dim)
        self.obs_dim = int(obs_dim)
        self.cond_predict_scale = bool(cond_predict_scale) and self.cond_dim > 0

        # obs normalizer (persistent, not in the reference checkpoints)
        if self.obs_dim > 0:
            self.obs_normalizer = ObsNormalizer(self.obs_dim)

        # channel ladder: [transition_dim, dim*m0, dim*m1, ...]
        dims = [self.transition_dim, *[dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # embed_dim = time_dim when cond_dim==0 (frozen Janner path), else time_dim+cond_dim
        time_dim = dim
        embed_dim = time_dim + self.cond_dim
        predict_scale = self.cond_predict_scale

        # time_mlp: SinusoidalPosEmb(dim)[0] → Linear(dim,dim*4)[1] → Mish[2] → Linear(dim*4,dim)[3]
        # SinusoidalPosEmb has only a non-persistent buffer → NOT in state_dict
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # encoder: ModuleList of [RTB, RTB, Downsample|Identity] per resolution
        self.encoder_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.encoder_blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim, cond_predict_scale=predict_scale),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim, cond_predict_scale=predict_scale),
                nn.Identity() if is_last else Downsample1d(dim_out),
            ]))

        mid_dim = dims[-1]
        self.middle_block_1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, cond_predict_scale=predict_scale)
        self.middle_block_2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, cond_predict_scale=predict_scale)

        # decoder: symmetric to encoder, using reversed(in_out[1:])
        self.decoder_blocks = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.decoder_blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, cond_predict_scale=predict_scale),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim, cond_predict_scale=predict_scale),
                nn.Identity() if is_last else Upsample1d(dim_in),
            ]))

        self.output_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, self.transition_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise a trajectory batch.

        Args:
            x:    (B, H, transition_dim) noisy trajectory.
            t:    (B,) per-sample timestep.
            cond: (B, cond_dim) global conditioning vector, or None.

        Returns:
            (B, H, transition_dim) denoised output.
        """
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        # channels-first for Conv1d
        h = x.permute(0, 2, 1).contiguous()  # (B, transition_dim, H)

        t_emb = self.time_mlp(t)  # (B, dim)

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond_dim>0 but cond=None")
            g = cond
            if self.obs_dim > 0:
                # apply obs normalizer to the obs slice of cond (Invariant 9)
                g = torch.cat([self.obs_normalizer(cond[:, :self.obs_dim]),
                                cond[:, self.obs_dim:]], dim=-1)
            t_emb = torch.cat([t_emb, g], dim=-1)  # (B, dim+cond_dim)

        skips = []
        for resnet, resnet2, downsample in self.encoder_blocks:
            h = resnet(h, t_emb)
            h = resnet2(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.middle_block_1(h, t_emb)
        h = self.middle_block_2(h, t_emb)

        for resnet, resnet2, upsample in self.decoder_blocks:
            h = torch.cat((h, skips.pop()), dim=1)
            h = resnet(h, t_emb)
            h = resnet2(h, t_emb)
            h = upsample(h)

        h = self.output_conv(h)
        return h.permute(0, 2, 1).contiguous()  # (B, H, transition_dim)

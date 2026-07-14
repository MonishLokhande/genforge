"""Temporal U-Net with optional linear self-attention — forge port of the reference implementation's
``temporal_unet_attn``, checkpoint-compatible with the trained locomotion planners.

Layout is the 4-slot-per-stage Janner variant: ``(ResBlock, ResBlock, Attn-or-Identity,
Down/Upsample)``. With ``attention=False`` (all six trained locomotion checkpoints) the
attention slots are parameter-free identities, so the only state_dict difference from
forge's 3-slot ``temporal_unet_janner`` is the sample conv living at slot index 3
instead of 2. Block internals are REUSED from forge's Janner module — those are already
proven key-for-key identical to the reference implementation.

# ---------------------------------------------------------------------------#
# ⚠  CHECKPOINT PARITY — DO NOT RENAME                                       #
# Module and parameter names are frozen to the Janner reference so   #
# trained checkpoints strict-load. attention classes are verbatim copies.    #
# ---------------------------------------------------------------------------#
"""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.core.interfaces import Model
from forge.core.registry import register
from forge.models.temporal_unet_janner import (
    Conv1dBlock,
    Downsample1d,
    ResidualTemporalBlock,
    Upsample1d,
    _SinusoidalPosEmb,
)


class ChannelLayerNorm(nn.Module):
    """Channel-first LayerNorm over (B, C, L) with per-channel affine."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = ChannelLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class LinearAttention(nn.Module):
    """Linear-complexity multi-head self-attention over the time axis (Shen et al. 2021)."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            t.reshape(t.shape[0], self.heads, t.shape[1] // self.heads, t.shape[-1])
            for t in qkv
        )
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[-1])
        return self.to_out(out)


def attention_slot(dim: int, attention: bool) -> nn.Module:
    return Residual(PreNorm(dim, LinearAttention(dim))) if attention else nn.Identity()


@register("model", "attention_unet")
class AttentionUNet(Model):
    """4-slot temporal U-Net (attention slots per stage), reference-checkpoint compatible.

    forward(x: (B, H, D), t: (B,), cond=None) -> (B, H, D). Unconditional only —
    the trained locomotion planners condition at sample time via Pin, not model cond.
    """

    output_type: str

    def __init__(
        self,
        *,
        horizon: int,
        transition_dim: int,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        attention: bool = False,
        output_type: str = "x0",
        cond_dim: int = 0,
    ):
        super().__init__()
        if cond_dim:
            # NOTE: unconditional only; add janner-style FiLM concat if a conditional
            # attention checkpoint ever exists.
            raise ValueError("attention_unet is unconditional (cond_dim must be 0)")
        self.output_type = output_type
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.dim = int(dim)
        self.dim_mults = tuple(int(m) for m in dim_mults)
        self.attention = bool(attention)

        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        time_dim = self.dim

        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(self.dim),
            nn.Linear(self.dim, self.dim * 4),
            nn.Mish(),
            nn.Linear(self.dim * 4, self.dim),
        )

        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.encoder_blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                attention_slot(dim_out, self.attention),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.middle_block_1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.middle_attention = attention_slot(mid_dim, self.attention)
        self.middle_block_2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.decoder_blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                attention_slot(dim_in, self.attention),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.output_conv = nn.Sequential(
            Conv1dBlock(self.dim, self.dim, kernel_size=5),
            nn.Conv1d(self.dim, self.transition_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.horizon or x.shape[2] != self.transition_dim:
            raise ValueError(
                f"expected x of shape [B, {self.horizon}, {self.transition_dim}], got {tuple(x.shape)}"
            )
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        h = x.permute(0, 2, 1).contiguous()
        t_emb = self.time_mlp(t)

        skips = []
        for resnet, resnet2, attn, downsample in self.encoder_blocks:
            h = resnet(h, t_emb)
            h = resnet2(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = downsample(h)

        h = self.middle_block_1(h, t_emb)
        h = self.middle_attention(h)
        h = self.middle_block_2(h, t_emb)

        for resnet, resnet2, attn, upsample in self.decoder_blocks:
            h = torch.cat((h, skips.pop()), dim=1)
            h = resnet(h, t_emb)
            h = resnet2(h, t_emb)
            h = attn(h)
            h = upsample(h)

        h = self.output_conv(h)
        return h.permute(0, 2, 1).contiguous()

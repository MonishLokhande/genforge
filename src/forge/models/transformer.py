"""A minimal sequence transformer — the SHARED backbone for the discrete-diffusion LMs.

Token + learned-positional embedding, a stack of bidirectional self-attention blocks (denoising
sees the whole corrupted sequence, so no causal mask), additive time conditioning, and a per-token
head over the vocabulary. ONE backbone, parameterized by config; `output_type` is config-selectable
so the same model serves x₀-logits (D3PM/MDLM) and SEDD's concrete-score parameterization. The head
always emits ``(B, L, V)``; what those values *mean* is the method's concern (Invariant 3-style).

Attention is written out with explicit ``q``/``k``/``v``/``out_proj`` ``nn.Linear`` modules (not
``nn.MultiheadAttention``) so every projection is a real submodule call — this is what makes the
whole block reachable by the generic LoRA injector (``utils.lora``) and avoids the fused-kernel
eval path that reads packed weights as raw tensors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.interfaces import Model
from ..core.registry import register
from .mlp import sinusoidal_embedding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE). Buffers are non-persistent — they're derived from config,
    not learned, so they stay out of the checkpoint and rebuild on load."""

    def __init__(self, dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))   # (dim/2,)
        emb = torch.cat([torch.outer(torch.arange(max_len).float(), inv_freq)] * 2, dim=-1)  # (L, dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:length], self.sin[:length]      # each (L, dim)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with separate q/k/v/out projections."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q(x).view(B, L, H, dh).transpose(1, 2)   # (B, H, L, dh)
        k = self.k(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v(x).view(B, L, H, dh).transpose(1, 2)
        if rope is not None:                              # rotate q,k in place of a learned pos table
            cos, sin = (r.to(q.dtype) for r in rope)      # (L, dh)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, L, D)            # (B, L, D)
        return self.out_proj(o)


class Block(nn.Module):
    """Pre-norm transformer block: attention + GELU MLP, each with a residual."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.linear2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), rope=rope))
        x = x + self.drop(self.linear2(self.drop(self.act(self.linear1(self.norm2(x))))))
        return x


@register("model", "transformer")
class Transformer(Model):
    def __init__(
        self,
        vocab_size: int = 32,
        length: int = 64,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        output_type: str = "logits",
        time_embed_dim: int = 32,
        dropout: float = 0.0,
        time_conditioned: bool = True,
        rope: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.length = length
        self.output_type = output_type
        self.time_embed_dim = time_embed_dim

        self.tok = nn.Embedding(vocab_size, d_model)
        # RoPE (rotary, carried into attention) replaces the learned positional table; the time MLP
        # is dropped entirely when time_conditioned=False (MDLM paper Table 12: the time-independent
        # net matches the time-conditioned one in perplexity at ~2x faster sampling).
        self.rope = RotaryEmbedding(d_model // n_heads, length) if rope else None
        self.pos = None if rope else nn.Parameter(torch.zeros(1, length, d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        ) if time_conditioned else None
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        if self.pos is not None:
            nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.tok(x)                                                       # (B, L, d_model)
        if self.pos is not None:
            h = h + self.pos[:, : x.shape[1]]
        if self.time_mlp is not None:
            t = torch.as_tensor(t, device=x.device)
            if t.ndim == 0:
                t = t.expand(x.shape[0])
            h = h + self.time_mlp(sinusoidal_embedding(t, self.time_embed_dim)).unsqueeze(1)
        rope = self.rope(x.shape[1]) if self.rope is not None else None
        for blk in self.blocks:
            h = blk(h, rope=rope)                                             # bidirectional attention
        return self.head(self.norm(h))                                       # (B, L, V)

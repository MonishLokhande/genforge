"""A dependency-light character-level text source — the SHARED text adapter for the discrete LMs.

Ships a tiny built-in public-domain corpus (Shakespeare's Sonnet 18), char-tokenizes it, and serves
fixed-length windows. The vocabulary is the corpus's unique characters plus one **mask** token (the
last index) so absorbing diffusion has somewhere to send noise. `evaluate` is the ONE shared LM
metric: the bits-per-character of generated samples under a corpus bigram model (lower = more
corpus-like), with the corpus's own bigram bits/char as the reference floor — sampler-agnostic, so
D3PM, MDLM and SEDD all use it. Heavy real tokenizers/corpora belong behind the `text` extra.
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core.registry import register

# Public-domain. Kept small and structured so a toy LM can learn its char statistics.
_SONNET = (
    "shall i compare thee to a summer's day?\n"
    "thou art more lovely and more temperate:\n"
    "rough winds do shake the darling buds of may,\n"
    "and summer's lease hath all too short a date:\n"
    "sometime too hot the eye of heaven shines,\n"
    "and often is his gold complexion dimm'd;\n"
    "and every fair from fair sometime declines,\n"
    "by chance or nature's changing course untrimm'd;\n"
    "but thy eternal summer shall not fade,\n"
    "nor lose possession of that fair thou ow'st;\n"
    "nor shall death brag thou wander'st in his shade,\n"
    "when in eternal lines to time thou grow'st:\n"
    "so long as men can breathe or eyes can see,\n"
    "so long lives this, and this gives life to thee.\n"
)


@register("environment", "char_text")
class CharText:
    def __init__(self, length: int = 64, repeat: int = 8, corpus: Optional[str] = None):
        text = (corpus if corpus is not None else _SONNET) * repeat
        self.length = int(length)
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.num_chars = len(self.chars)
        self.mask_index = self.num_chars                      # absorbing mask = extra last index
        self.vocab_size = self.num_chars + 1
        self._ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self._bigram = self._fit_bigram(self._ids)            # (V_data, V_data) P(next | cur)
        self._ref_bpc = self._bits_per_char(self._ids.unsqueeze(0))

    @property
    def dim(self) -> int:
        return self.length

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.chars[int(i)] for i in ids if int(i) < self.num_chars)

    def visualize(self, samples: torch.Tensor, out: str) -> str:
        from ..render import write_transcript
        return write_transcript(samples, self.decode, out)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        hi = self._ids.shape[0] - self.length
        starts = torch.randint(0, hi, (n,), generator=generator)
        return self._ids[starts.unsqueeze(1) + torch.arange(self.length)]   # (n, length)

    # ── shared evaluator: bigram bits-per-char ──────────────────────────────────────────────────
    def _fit_bigram(self, ids: torch.Tensor) -> torch.Tensor:
        v = self.num_chars
        counts = torch.ones(v, v)                             # Laplace smoothing
        counts.index_put_((ids[:-1], ids[1:]), torch.ones(ids.shape[0] - 1), accumulate=True)
        return counts / counts.sum(dim=-1, keepdim=True)

    def _bits_per_char(self, seqs: torch.Tensor) -> float:
        s = seqs.clamp(max=self.num_chars - 1)               # any surviving mask → last data char
        p = self._bigram.to(s.device)[s[:, :-1], s[:, 1:]].clamp_min(1e-12)
        return float((-torch.log2(p)).mean().item())

    def evaluate(self, samples: torch.Tensor) -> dict:
        return {
            "bits_per_char": self._bits_per_char(samples),
            "ref_bits_per_char": self._ref_bpc,
            "n": float(samples.shape[0]),
        }

"""TinyStories adapter — the REAL GPT-2 BPE tokenizer over the REAL roneneldan/TinyStories corpus.

A small, dependency-light-by-default LM source: streams a `max_stories` slice (no full download),
tokenizes with the GPT-2 BPE (`tiktoken`, vocab 50257, EOT=50256), EOT-joins stories, packs into
fixed `ctx` windows, and reserves ``[MASK]=50257`` so the absorbing graph has a noise sink
(vocab=50258). The tokenized slice is cached to disk (reruns don't re-stream/re-tokenize).

Heavy deps (`tiktoken`, `datasets`) live ONLY in the `text` uv extra and are imported lazily — core
`uv sync` works without them; a missing extra raises a clear install hint. The evaluator is a
**unigram** bits/token (a bigram table at V=50257 would itself be ~10 GB); it measures token-frequency
realism vs. uniform-random — a small-budget "beats random" signal, not coherence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch

from forge.core.registry import register


@register("environment", "tinystories")
class TinyStories:
    EOT = 50256          # GPT-2 end-of-text
    MASK = 50257         # injected absorbing mask token
    VOCAB = 50258        # 50257 GPT-2 tokens (incl EOT) + [MASK]
    NUM_DATA = 50257     # data-token vocabulary (0..50256); mask is never in the data

    def __init__(
        self,
        ctx: int = 256,
        max_stories: int = 4000,
        split: str = "train",
        cache_dir: str = ".cache/tinystories",
        seed: int = 0,
    ):
        self.ctx = int(ctx)
        self.max_stories = int(max_stories)
        self.split = split
        self.cache_dir = cache_dir
        self.seed = int(seed)

        self._data = self._load_or_build()                       # (N, ctx) long, data tokens only
        counts = torch.bincount(self._data.reshape(-1), minlength=self.NUM_DATA).float()
        unigram = (counts + 1.0) / (counts.sum() + self.NUM_DATA)  # Laplace-smoothed
        self._log2_unigram = -torch.log2(unigram)
        self._ref_bpt = float((unigram * self._log2_unigram).sum().item())  # corpus unigram entropy

    # ── shared env interface ────────────────────────────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self.ctx

    @property
    def vocab_size(self) -> int:
        return self.VOCAB

    @property
    def mask_index(self) -> int:
        return self.MASK

    @property
    def length(self) -> int:
        return self.ctx

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.randint(0, self._data.shape[0], (n,), generator=generator)
        return self._data[idx]

    def decode(self, ids: torch.Tensor) -> str:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.decode([int(i) for i in ids if int(i) < self.NUM_DATA])

    def visualize(self, samples: torch.Tensor, out: str) -> str:
        from ..render import write_transcript
        return write_transcript(samples, self.decode, out)

    def evaluate(self, samples: torch.Tensor) -> dict:
        s = samples.clamp(max=self.NUM_DATA - 1).reshape(-1)
        return {
            "bits_per_token": float((self._log2_unigram.to(s.device)[s]).mean().item()),
            "ref_bits_per_token": self._ref_bpt,
            "random_bits_per_token": math.log2(self.NUM_DATA),
            "n": float(samples.numel()),
        }

    # ── tokenize / pack (pure logic is offline-testable; streaming is lazy) ──────────────────────
    @staticmethod
    def pack(token_lists: List[List[int]], ctx: int, eot: int) -> torch.Tensor:
        """EOT-join stories and chunk into non-overlapping ``ctx`` windows → ``(N, ctx)`` long."""
        flat: List[int] = []
        for toks in token_lists:
            flat.extend(toks)
            flat.append(eot)
        n = (len(flat) // ctx) * ctx
        if n == 0:
            raise ValueError(f"only {len(flat)} tokens — not enough to fill one ctx={ctx} window.")
        return torch.tensor(flat[:n], dtype=torch.long).reshape(-1, ctx)

    def _stream_tokenize(self) -> List[List[int]]:
        try:
            import tiktoken
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - only without the text extra
            raise ImportError(
                "TinyStories needs the `text` extra: `uv sync --extra text` (tiktoken + datasets)."
            ) from e
        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
        lists: List[List[int]] = []
        for i, ex in enumerate(ds):
            if i >= self.max_stories:
                break
            lists.append(enc.encode(ex["text"]))
        return lists

    def _load_or_build(self) -> torch.Tensor:
        cache = Path(self.cache_dir) / f"{self.split}_{self.max_stories}_{self.ctx}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=False)
        data = self.pack(self._stream_tokenize(), self.ctx, self.EOT)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
        return data

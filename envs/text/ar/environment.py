"""GPT-2 BPE corpus for the autoregressive LM, with an HONEST contiguous train/val split.

Tokenizes with the GPT-2 BPE (`tiktoken`, vocab 50257, EOS=50256) — the same tokenizer the diffusion
`tinystories` env uses — and holds a **disjoint** token split: the train region and the val region
share no tokens, so a train window can never overlap a val window (a random index split leaks,
because LM windows overlap by `block_size`). `decode`/`visualize` let the reused `env_render`
visualizer write generated tokens to a readable transcript.

`source="builtin"` BPE-encodes a small built-in passage (offline, for CI/smoke — needs only the
cached tiktoken vocab). `source="tinystories"` streams roneneldan/TinyStories and EOS-brackets each
story (needs the `text` extra: `tiktoken` + `datasets`). Vocab is fixed by the tokenizer (50257),
independent of the corpus.
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core.registry import register


def _require_tiktoken():
    try:
        import tiktoken
    except ModuleNotFoundError as e:                          # heavy dep lives in the `text` extra
        raise ModuleNotFoundError(
            "ar_text needs the GPT-2 BPE tokenizer — install the text extra: "
            "`uv sync --extra text` (adds tiktoken/datasets)."
        ) from e
    return tiktoken.get_encoding("gpt2")


@register("environment", "ar_text")
class ARText:
    def __init__(self, source: str = "builtin", n_stories: int = 2000, repeat: int = 200,
                 val_frac: float = 0.1, corpus: Optional[str] = None,
                 cache_dir: str = ".cache/ar_text"):
        self.enc = _require_tiktoken()
        self.eos_id = self.enc.eot_token                      # 50256
        self.vocab_size = self.enc.n_vocab                    # 50257 — fixed by the tokenizer
        self.cache_dir = cache_dir

        if corpus is not None:
            ids = self.enc.encode(corpus)
        elif source == "builtin":
            from ..char.environment import _SONNET            # reuse the public-domain built-in
            doc = self.enc.encode(_SONNET)
            ids = []
            for _ in range(repeat):
                ids.extend(doc)
                ids.append(self.eos_id)                       # EOS-bracket each copy, like tinystories,
                                                              # so the BOS seed token (50256) is trained
        elif source == "tinystories":
            ids = self._stream_tinystories(n_stories)
        else:
            raise ValueError(f"unknown source {source!r}; use 'builtin' or 'tinystories'.")

        ids = ids if torch.is_tensor(ids) else torch.tensor(ids, dtype=torch.long)
        n = int(len(ids) * (1.0 - float(val_frac)))           # boundary; the two regions are disjoint
        self.train_tokens = ids[:n]
        self.val_tokens = ids[n:]

    def _stream_tinystories(self, n_stories: int) -> torch.Tensor:
        from pathlib import Path

        cache = Path(self.cache_dir) / f"tokens_{n_stories}.pt"   # cache the tokenized slice; reruns skip re-streaming
        if cache.exists():
            return torch.load(cache)
        try:
            import datasets
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "source='tinystories' needs `datasets` — install the text extra: "
                "`uv sync --extra text`."
            ) from e
        datasets.disable_progress_bars()
        ds = datasets.load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        toks: list = []
        for i, ex in enumerate(ds):
            toks.extend(self.enc.encode(ex["text"].strip()))
            toks.append(self.eos_id)                          # bracket every story with EOS
            if i + 1 >= n_stories:
                break
        out = torch.tensor(toks, dtype=torch.long)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
        return out

    def region(self, split: str) -> torch.Tensor:
        assert split in ("train", "val"), split
        return self.train_tokens if split == "train" else self.val_tokens

    def decode(self, ids: torch.Tensor) -> str:
        return self.enc.decode([int(i) for i in ids if int(i) != self.eos_id])   # hide the boundary token

    def visualize(self, samples: torch.Tensor, out: str) -> str:
        from ..render import write_transcript
        return write_transcript(samples, self.decode, out)

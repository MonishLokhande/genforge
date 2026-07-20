"""LM windows over one disjoint region of the AR corpus. `gather` is pure in idx (resumable, Inv 5).

The region (train / val) comes from the injected `ar_text` environment, so the corpus and its honest
split live in ONE place. `split="train"` trains only on the train region — the model never sees a val
token, which is what makes any later held-out estimate honest.
"""

from __future__ import annotations

import torch

from forge.core.protocols import BaseDataset
from forge.core.registry import register


@register("dataset", "ar_windows")
class ARWindows(BaseDataset):
    def __init__(self, environment=None, block_size: int = 64, split: str = "train"):
        assert environment is not None, (
            "ar_windows needs the ar_text environment — set `plugins: [envs.text.ar]` and "
            "`environment: {name: ar_text}` in the experiment."
        )
        self.env = environment
        self.block_size = int(block_size)
        self.seq_len = self.block_size + 1              # x = window[:-1], y = window[1:]
        self.split = split
        self.data = environment.region(split)
        assert self.data.numel() > self.seq_len, (
            f"corpus {split!r} region has {self.data.numel()} tokens < one window ({self.seq_len}); "
            f"use a longer corpus (a larger environment.params.repeat, or source=tinystories) or a "
            f"smaller block_size."
        )

    @property
    def fit_tensor(self) -> torch.Tensor:              # unused (no preprocessor membrane for tokens)
        return self.data[: self.seq_len].reshape(1, -1).float()

    @property
    def num_items(self) -> int:
        return self.data.numel() - self.seq_len + 1    # last window ends at the final token

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (self.seq_len,)

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to("cpu")                            # index the CPU corpus even if the runner is on GPU
        offsets = torch.arange(self.seq_len)
        return self.data[idx.reshape(-1, 1) + offsets.reshape(1, -1)].to(torch.int64)

    def decode(self, ids) -> str:
        return self.env.decode(ids)

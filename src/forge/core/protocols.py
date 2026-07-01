"""Env-side data contracts — the *plugin surface*, DISTINCT from the membrane ABCs in ``interfaces.py``.

``interfaces.py`` owns the framework membrane (``Space`` / ``Schedule`` / ``Model`` / ``Method`` /
``Sampler`` / ``Cost`` / ``Controller`` / ``Preprocessor`` / ``Runner``). This module owns the
contracts that concrete ``envs/*`` plugin packages implement on the **data-source** side of that
membrane:

  - :class:`BatchProtocol` — the shape of one training batch as it ENTERS the loop, in raw units,
    *before* the preprocessor membrane.
  - :class:`BaseDataset` — what the training runner programs against to pull batches.
  - :class:`BaseProcessor` — env-specific encoding that happens BEFORE the membrane (tokenization,
    packing, windowing).

These are deliberately NOT in ``interfaces.py``: they are the plugin boundary, not the core
membrane, and concrete ``envs/*`` packages — never ``src/forge/`` — implement them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class BatchProtocol:
    """One training batch as it enters the loop, in RAW units, BEFORE the preprocessor membrane.

    The training runner consumes exactly these fields (Step-1 audit of ``runners/training.py``):
    it reads ``x0``, passes ``cond`` straight through to ``method.loss`` (never through the
    preprocessor — conditioning is never normalized, Invariant 9), and may consult ``mask``.

    Fields
    ------
    x0:
        ``(B, *state_shape)`` — the generated quantity, raw units. The leading axis is the batch;
        the rest is the per-item ``sample_shape`` and is **not** fixed to a rank, because genforge
        envs have incompatible shapes:
          - distributions: ``(B, D)``    float32
          - trajectories:  ``(B, T, D)`` float32
          - sequences:     ``(B, L)``    int64   (discrete token ids)
    cond:
        ``(B, *cond_shape)`` optional conditioning (may be ``None``). Passed to ``method.loss``;
        NEVER passed through ``preprocessor.transform`` (Invariant 9).
    mask:
        ``(B, T)`` bool padding / inpaint mask (may be ``None``).
    """

    x0: torch.Tensor
    cond: Optional[Any] = None
    mask: Optional[torch.Tensor] = None


class BaseDataset(abc.ABC):
    """The data-source contract the training runner and planner program against.

    Concrete env datasets implement the *raw tensor* surface the loop already relies on
    (``gather`` / ``fit_tensor`` / ``num_items`` / ``sample_shape``) plus a ``dim`` attribute.
    :meth:`batch` is the :class:`BatchProtocol` entry point the runner uses; its default simply
    wraps :meth:`gather`, so adopting this base is a structural move, not a per-env rewrite. A
    dataset that carries conditioning / masks overrides :meth:`batch` to attach them.
    """

    # ── raw tensor surface (load-bearing for the runner + planner) ──────────────────────────────
    @property
    @abc.abstractmethod
    def fit_tensor(self) -> torch.Tensor:
        """The tensor the preprocessor membrane fits on (per-feature stats)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_items(self) -> int:
        """Number of indexable items the loop samples from."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sample_shape(self) -> tuple[int, ...]:
        """Per-item shape (without the batch axis), e.g. ``(D,)`` or ``(H, D)`` or ``(L,)``."""
        raise NotImplementedError

    @abc.abstractmethod
    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Batched gather → ``(B, *sample_shape)`` in raw units / raw dtype (float32 | int64)."""
        raise NotImplementedError

    # ── BatchProtocol surface (additive; the runner's entry point) ──────────────────────────────
    def batch(self, idx: torch.Tensor) -> BatchProtocol:
        """Wrap a raw :meth:`gather` into a :class:`BatchProtocol`. Override to attach cond/mask."""
        return BatchProtocol(x0=self.gather(idx))

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> BatchProtocol:
        """Single-item :class:`BatchProtocol` (``idx`` is a python int)."""
        return BatchProtocol(x0=self.gather(torch.as_tensor([idx])).squeeze(0))

    @classmethod
    def validate_batch(cls, batch: BatchProtocol) -> None:
        """Assert a batch satisfies the contract before it crosses the preprocessor membrane.

        Fails loudly: ``x0`` must be a float32 (continuous) or int64 (discrete)
        Tensor; ``cond``'s batch dim must match ``x0`` when present; ``mask`` must be bool.
        """
        if not isinstance(batch, BatchProtocol):
            raise TypeError(f"expected BatchProtocol, got {type(batch).__name__}.")
        if not torch.is_tensor(batch.x0):
            raise TypeError(f"BatchProtocol.x0 must be a Tensor, got {type(batch.x0).__name__}.")
        if batch.x0.dtype not in (torch.float32, torch.int64):
            raise TypeError(
                f"BatchProtocol.x0 must be float32 (continuous) or int64 (discrete), "
                f"got {batch.x0.dtype}."
            )
        if batch.cond is not None and torch.is_tensor(batch.cond):
            if batch.cond.shape[0] != batch.x0.shape[0]:
                raise ValueError(
                    f"BatchProtocol.cond batch dim {batch.cond.shape[0]} != x0 batch dim "
                    f"{batch.x0.shape[0]}."
                )
        if batch.mask is not None and batch.mask.dtype != torch.bool:
            raise TypeError(f"BatchProtocol.mask must be bool, got {batch.mask.dtype}.")


class BaseProcessor(abc.ABC):
    """Env-specific PRE-membrane encoding — DISTINCT from the membrane ``Preprocessor``.

    ``interfaces.Preprocessor`` is the affine **membrane** that operates in normalized space
    (``fit`` / ``transform`` / ``inverse``; stats travel in the checkpoint, Invariants 2/8). A
    ``BaseProcessor`` is the OUTER, env-owned encoding that turns raw observations into model-space
    tensors and wraps them in a :class:`BatchProtocol` — it runs OUTSIDE and BEFORE the membrane and
    carries **no** normalization statistics. Examples: BPE tokenization, story packing, trajectory
    windowing. Do NOT merge the two: a ``BaseProcessor`` never standardizes; a ``Preprocessor``
    never tokenizes. Conditioning passes through unchanged (Invariant 9).
    """

    @abc.abstractmethod
    def process(self, raw: Any) -> BatchProtocol:
        """Encode raw env output into a model-space :class:`BatchProtocol` (tokenize / pack / window)."""
        raise NotImplementedError

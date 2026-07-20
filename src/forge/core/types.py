"""Shared dataclasses passed between components.

Kept deliberately small in Phase 0 — the load-bearing types (`Trajectory`, `SamplerOutput`) and
`Provenance`, which the self-contained checkpoint embeds (Invariant 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class Provenance:
    """Reproducibility stamp embedded in every checkpoint and W&B run."""

    git_hash: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class SamplerOutput:
    """Result of a `Sampler.sample` call.

    `samples` are the final iterates; `chain` (optional) is the full reverse path when the caller
    requested `return_chain=True`. Both are in normalized coordinates — the runner inverts.
    """

    samples: "Tensor"
    chain: "Optional[Tensor]" = None

"""Pre-membrane processor for the synthetic-trajectory env.

The real pre-membrane encoding — flat-tensor windowing of episodes into ``(B, H, dim)`` windows —
already lives in ``TrajectoryDataset.gather`` (the ≈90× memory pattern), so this processor is an
identity wrapper into a :class:`BatchProtocol` over an already-gathered window. It is the per-package
home for a future standalone windowing encoder. Distinct from the trajectory MEMBRANE preprocessors
in ``membrane.py`` (those standardize; a ``BaseProcessor`` never does — ``forge.core.protocols``).
"""

from __future__ import annotations

from typing import Any

from forge.core.protocols import BaseProcessor, BatchProtocol


class WindowingProcessor(BaseProcessor):
    def process(self, raw: Any) -> BatchProtocol:
        return BatchProtocol(x0=raw)

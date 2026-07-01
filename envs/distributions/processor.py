"""Pre-membrane processor for the 2-D distribution envs.

These envs already emit model-space float tensors from ``Environment.sample`` (no tokenization or
packing), so the processor is an identity pass-through that wraps a raw batch into a
:class:`BatchProtocol`. Normalization is the MEMBRANE's job (``standardize`` / ``minmax``), NOT this
processor's — see the ``BaseProcessor`` vs ``Preprocessor`` distinction in
``forge.core.protocols``. (Documented placeholder: a real pre-membrane encoder would live here.)
"""

from __future__ import annotations

from typing import Any

from forge.core.protocols import BaseProcessor, BatchProtocol


class IdentityProcessor(BaseProcessor):
    def process(self, raw: Any) -> BatchProtocol:
        return BatchProtocol(x0=raw)

"""Pre-membrane processor for the categorical-toy env.

The env emits model-space int64 token ids directly from ``Environment.sample`` — no tokenization
step — so the processor is an identity wrapper into a :class:`BatchProtocol`. (Documented
placeholder; see ``BaseProcessor`` vs ``Preprocessor`` in ``forge.core.protocols``.)
"""

from __future__ import annotations

from typing import Any

from forge.core.protocols import BaseProcessor, BatchProtocol


class IdentityProcessor(BaseProcessor):
    def process(self, raw: Any) -> BatchProtocol:
        return BatchProtocol(x0=raw)

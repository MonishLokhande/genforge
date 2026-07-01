"""Pre-membrane processor for the TinyStories env.

The real pre-membrane encoding (GPT-2 BPE tokenization + EOT-join + ctx packing) already lives in
``TinyStories`` (``TinyStories.pack`` / ``_stream_tokenize``), so this processor is an identity
wrapper into a :class:`BatchProtocol`. Distinct from the membrane ``Preprocessor`` — a
``BaseProcessor`` tokenizes/packs; it never standardizes (``forge.core.protocols``).
"""

from __future__ import annotations

from typing import Any

from forge.core.protocols import BaseProcessor, BatchProtocol


class TinyStoriesProcessor(BaseProcessor):
    def process(self, raw: Any) -> BatchProtocol:
        return BatchProtocol(x0=raw)

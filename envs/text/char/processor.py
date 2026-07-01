"""Pre-membrane processor for the char-text env.

Char tokenization already happens inside ``CharText`` (``sample`` returns int64 token-id windows),
so this processor is an identity wrapper into a :class:`BatchProtocol`. It is the per-package home
for any future standalone pre-membrane text encoder; today it documents that the encoding lives in
the environment. Distinct from the membrane ``Preprocessor`` (``forge.core.protocols``).
"""

from __future__ import annotations

from typing import Any

from forge.core.protocols import BaseProcessor, BatchProtocol


class TextProcessor(BaseProcessor):
    def process(self, raw: Any) -> BatchProtocol:
        return BatchProtocol(x0=raw)

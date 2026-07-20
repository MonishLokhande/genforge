"""Shared text-rendering helper — the piece every text env reuses.

A text env's `visualize()` is a one-liner delegating here, so diLLM (d3pm/mdlm/sedd) and a future
ARLLM produce the *same* readable transcript from the *same* code. The env-specific part is only its
own `decode(ids) -> str`; this decides the layout.
"""

from __future__ import annotations

from pathlib import Path


def write_transcript(samples, decode, out, sep="\n\n---\n\n", max_n=None) -> str:
    """Decode each sample row to a string and write the batch to `out`. Returns the path.

    `samples` is `(N, length)` token ids; `decode` is the env's `decode`. `max_n` caps how many are
    written (a 256-sample eval batch does not need 256 blocks on disk); None writes all.
    """
    rows = samples if max_n is None else samples[:max_n]
    blocks = [decode(row) for row in rows]
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sep.join(blocks), encoding="utf-8")
    return str(p)

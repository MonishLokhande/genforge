"""Persist generated samples (`.npz`) and evaluation metrics (`.json`) to a run's output dir.

Deliberately NOT the checkpoint schema (`core/checkpoint.py`) — samples/metrics are lightweight
artifacts, not resumable state. Metric keys stay a flat ``{str: float}`` (aggregation depends on it);
`save_metrics` stamps a `"step"` sidecar so a stale/mismatched file is visible, not silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def save_samples(out_dir: str, x: torch.Tensor, name: str = "samples.npz") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    np.savez(p, samples=x.detach().cpu().numpy())
    return str(p)


def load_samples(path: str) -> torch.Tensor:
    """Load samples written by `save_samples` (accepts the dir or the .npz file path)."""
    p = Path(path)
    if p.is_dir():
        p = p / "samples.npz"
    with np.load(p) as d:
        return torch.from_numpy(d["samples"])


def save_metrics(out_dir: str, metrics: dict, step=None, name: str = "metrics.json") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    if step is not None:
        payload["step"] = int(step)
    p = out / name
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(p)

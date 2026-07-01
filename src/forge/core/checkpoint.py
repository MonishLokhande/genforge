"""Self-contained checkpoint format (Invariant 5).

Every checkpoint embeds everything needed to (a) `sample` from the `.pt` alone and (b) resume
bit-identically: model weights, EMA shadow, fitted preprocessor stats, the resolved config, and
provenance (git hash + seed), plus optimizer/scheduler/RNG state.

The *shape* of the dict is fixed here so it never has to change.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .types import Provenance

# Bump when the on-disk layout changes incompatibly.
CHECKPOINT_FORMAT_VERSION = 1


def current_git_hash() -> Optional[str]:
    """Best-effort current commit hash for provenance. Returns None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_checkpoint(
    *,
    model_state: Optional[dict] = None,
    ema_state: Optional[dict] = None,
    preprocessor_state: Optional[dict] = None,
    config: Any = None,
    provenance: Optional[Provenance] = None,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    rng_state: Optional[dict] = None,
    lora_config: Optional[dict] = None,
    step: int = 0,
) -> dict:
    """Assemble the self-contained checkpoint dict (Invariant 5).

    Every field is present so a reader never has to guess; unset fields are explicit ``None``.

    ``lora_config`` is the explicit contract for reconstructing adapter modules at load time: a
    LoRA checkpoint's keys (``...base.weight`` / ``.A`` / ``.B``) only load into a model that has
    had ``apply_lora`` run with the *same* ``r``/``alpha``/``targets``, so those params must travel
    with the weights rather than depend on the embedded Hydra config being present.
    """
    if provenance is None:
        provenance = Provenance(git_hash=current_git_hash())
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model_state": model_state,
        "ema_state": ema_state,
        "preprocessor_state": preprocessor_state,
        "config": config,
        "provenance": asdict(provenance),
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": rng_state,
        "lora_config": lora_config,
    }


def save_checkpoint(path: str | Path, checkpoint: dict) -> Path:
    """Write a checkpoint dict to ``path`` (creating parent dirs). Returns the path."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dict from ``path``."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)

"""Global + per-worker seeding and opt-in determinism."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, PyTorch (CPU+CUDA). ``deterministic`` opts into reproducible kernels."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A seeded :class:`torch.Generator` on ``device`` (for reproducible sampling)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

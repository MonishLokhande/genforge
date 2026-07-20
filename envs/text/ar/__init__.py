"""Autoregressive GPT-2-BPE LM as a bundled forge plugin (a non-diffusion guest paradigm).

Importing this package fires the @register decorators for the AR environment, dataset, method, and
sampler. The model is the shared `transformer` (causal); the built-in `training` runner and the
`env_render` visualizer are reused as-is.
"""

from . import components  # noqa: F401  — registers method=autoregressive, sampler=autoregressive
from .dataset import ARWindows
from .environment import ARText

Environment = ARText
Dataset = ARWindows

__all__ = ["Environment", "Dataset", "ARText", "ARWindows"]

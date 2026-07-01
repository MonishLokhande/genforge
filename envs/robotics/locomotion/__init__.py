"""Locomotion env package. Importing registers both adapters and the shared windowing dataset."""
from .. import trajectory_window  # noqa: F401 — @register("dataset","trajectory_window")
from . import adapter_d4rl  # noqa: F401 — @register("environment","d4rl") + aliases
from . import adapter_minari  # noqa: F401 — @register("environment","minari") + aliases

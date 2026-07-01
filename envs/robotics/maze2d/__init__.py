"""Maze2D env package. Importing registers the adapter and the shared windowing dataset."""
from .. import trajectory_window  # noqa: F401 — @register("dataset","trajectory_window")
from . import adapter  # noqa: F401 — @register("environment","maze2d") + aliases

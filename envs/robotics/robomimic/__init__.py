"""Robomimic lowdim adapter (robosuite manipulation demos, HDF5).

Importing this package registers the ``robomimic`` environment adapter and the shared
``trajectory_window`` dataset, so an experiment's ``plugins:`` needs only this one entry.
All robosuite/robomimic/h5py imports are lazy (inside methods)."""

from . import adapter  # noqa: F401 — @register("environment", "robomimic")
from .. import trajectory_window  # noqa: F401 — @register("dataset", "trajectory_window")

"""Shared robotics env infrastructure: the family-agnostic TrajectoryWindowDataset.

Importing this package registers ``@register("dataset", "trajectory_window")``. Each robotics
family package (``envs/robotics/<family>/``) imports ``..trajectory_window`` from its own
``__init__`` so one ``plugins:`` entry registers the adapter AND the shared windowing dataset.
"""

from . import trajectory_window  # noqa: F401  — import for the @register side effect

__all__ = ["trajectory_window"]

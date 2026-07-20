"""The synthetic-trajectory env package (its own windowing dataset).

Importing this package registers the environment (``.environment``) and the windowing
``trajectory`` dataset (``.dataset``). Trajectory experiments use the built-in ``minmax`` /
``standardize`` membrane preprocessors directly.
"""

from .dataset import TrajectoryDataset as Dataset
from .environment import SyntheticTrajectories

Environment = SyntheticTrajectories

__all__ = ["Environment", "Dataset", "SyntheticTrajectories"]

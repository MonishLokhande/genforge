"""The char-level text env package (the shared discrete-LM source)."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import CharText

Environment = CharText

__all__ = ["Environment", "Dataset", "CharText"]

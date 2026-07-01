"""The char-level text env package (the shared discrete-LM source)."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import CharText
from .processor import TextProcessor as Processor

Environment = CharText

__all__ = ["Environment", "Dataset", "Processor", "CharText"]

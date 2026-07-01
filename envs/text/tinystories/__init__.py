"""The TinyStories (real GPT-2 BPE) env package."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import TinyStories
from .processor import TinyStoriesProcessor as Processor

Environment = TinyStories

__all__ = ["Environment", "Dataset", "Processor", "TinyStories"]

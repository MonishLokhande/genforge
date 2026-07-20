"""The TinyStories (real GPT-2 BPE) env package."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import TinyStories

Environment = TinyStories

__all__ = ["Environment", "Dataset", "TinyStories"]

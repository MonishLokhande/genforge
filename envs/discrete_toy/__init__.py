"""The categorical-toy discrete env package."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import CategoricalToy

Environment = CategoricalToy

__all__ = ["Environment", "Dataset", "CategoricalToy"]

"""The categorical-toy discrete env package."""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import CategoricalToy
from .processor import IdentityProcessor as Processor

Environment = CategoricalToy

__all__ = ["Environment", "Dataset", "Processor", "CategoricalToy"]

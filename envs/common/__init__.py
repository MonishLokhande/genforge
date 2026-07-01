"""Shared, env-agnostic data-source infrastructure.

``DistributionDataset`` ("sample N raw points from any environment") is used by every sampling
family — distributions, discrete_toy, text, tinystories — so it lives here rather than in any single
env package. The plugin loader imports ``envs.common`` first, so experiments declare only their true
env plugin.
"""

from .dataset import DistributionDataset

__all__ = ["DistributionDataset"]

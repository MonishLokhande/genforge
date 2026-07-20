"""The 2-D synthetic distributions env package (gaussian_mixture / two_moons / swiss_roll).

Importing this package fires the ``@register`` decorators for its environments (via ``.environment``)
and the shared ``distribution`` dataset (via ``envs.common``). Exports the package contract
``Environment`` / ``Dataset``.
"""

from envs.common.dataset import DistributionDataset as Dataset

from .environment import GaussianMixture2D, SwissRoll, TwoMoons

Environment = GaussianMixture2D

__all__ = ["Environment", "Dataset", "GaussianMixture2D", "TwoMoons", "SwissRoll"]

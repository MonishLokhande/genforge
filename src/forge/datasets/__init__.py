"""Reserved namespace — concrete datasets are NOT framework code.

The dataset CONTRACT is the abstract ``BaseDataset`` in ``forge.core.protocols``; concrete
datasets (``distribution``, ``trajectory``) live in the repo-root ``envs/`` plugin tree and register
via an experiment's ``plugins:`` field. ``src/forge/`` owns only the abstract/base code.
"""


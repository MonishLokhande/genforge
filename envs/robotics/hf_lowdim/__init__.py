"""HF lowdim adapter (LeRobot-format PushT / Aloha) — parquet-backed, lowdim only.

Importing this package registers the ``hf_lowdim`` environment adapter and the shared
``trajectory_window`` dataset (one ``plugins:`` entry covers both). The ``datasets`` /
``gymnasium`` / ``gym_pusht`` / ``gym_aloha`` imports are lazy (inside methods)."""

from . import adapter  # noqa: F401 — @register("environment", "hf_lowdim")
from .. import trajectory_window  # noqa: F401 — @register("dataset", "trajectory_window")

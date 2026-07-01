"""Hydra composition: turn CLI overrides into a resolved config the builder can consume.

The primary config dir is the package ``configs/`` (the per-category groups + root). The repo-root
directory that *contains* the ``experiment/`` tree is added to the Hydra searchpath via the
``GENFORGE_EXP_ROOT`` env var, which this module sets before composing. The ``experiment`` config
group is then ``<GENFORGE_EXP_ROOT>/experiment``, and experiments may reference bases by absolute
path (e.g. ``- /experiment/<env>/base``). This keeps experiments as repo-root base+delta bundles
without hard-coding absolute paths into the configs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import forge


def configs_dir() -> str:
    return str(Path(forge.__file__).parent / "configs")


def searchpath_root() -> Path:
    """The directory that CONTAINS the ``experiment/`` tree (env override → cwd → package-relative)."""
    env = os.environ.get("GENFORGE_EXP_ROOT")
    if env:
        return Path(env)
    if (Path.cwd() / "experiment").is_dir():
        return Path.cwd()
    # Package layout src/forge/__init__.py → parents[2] is the repo root.
    return Path(forge.__file__).resolve().parents[2]


def compose_config(overrides: Sequence[str]):
    """Compose the resolved ``DictConfig`` from CLI-style overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    os.environ["GENFORGE_EXP_ROOT"] = str(searchpath_root())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=configs_dir()):
        cfg = compose(config_name="config", overrides=list(overrides))
    return cfg

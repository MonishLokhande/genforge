"""Plugin loading — import the concrete ``envs/*`` packages an experiment declares.

Concrete environments live in the repo-root ``envs/`` tree (NOT in the installed ``genforge``
package), so they register via the experiment's ``plugins:`` field rather than the builder's
built-in module list (Invariant 7). This module:

  - puts the repo root on ``sys.path`` so ``import envs.*`` resolves (the package install only adds
    ``src/`` via its ``.pth`` — the repo root that contains ``envs/`` is not otherwise importable);
  - :func:`load_plugins` imports a declared list (used by the builder, from the resolved config);
  - :func:`load_bundled_envs` imports every shipped env (used by ``forge list`` and the test
    suite, which have no experiment / ``plugins:`` selected) so the full catalog still appears.

The loader lives under ``src/forge/`` (always importable) precisely to avoid a chicken-and-egg:
it must fix ``sys.path`` *before* the first ``import envs.*``.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib.metadata import entry_points
from typing import Iterable

# Standard-Python plugin group: any installed package can advertise forge components by declaring
#   [project.entry-points."forge.plugins"]
#   <name> = "<module_to_import>"
# in its pyproject. Discovery imports each such module (firing its @register decorators) so it
# appears in `forge list` and is usable at train/sample time — no edit to forge required.
FORGE_PLUGIN_GROUP = "forge.plugins"

# The bundled env packages (the one place that enumerates what ships in ``envs/``). ``envs.common``
# holds the generic, env-agnostic ``distribution`` dataset shared across the sampling families.
BUNDLED_ENVS: tuple[str, ...] = (
    "envs.common",
    "envs.distributions",
    "envs.discrete_toy",
    "envs.text",  # the text/LM family — registers char_text, tinystories, and the AR ar_text
    "envs.trajectory_synth",
    # Robotics families — one plugin unit each (experiments select the family, not the parent). Their
    # sim/IO deps (mujoco/minari/robosuite) are imported LAZILY, so importing the adapter just fires
    # @register and works WITHOUT the `robotics` extra; the catalog lists them regardless.
    "envs.robotics.locomotion",
    "envs.robotics.maze2d",
    "envs.robotics.robomimic",
    "envs.robotics.hf_lowdim",
)


def _ensure_repo_root_on_path() -> None:
    """Prepend the repo root (the dir that contains ``envs/`` and ``experiment/``) to ``sys.path``."""
    from .compose import searchpath_root

    root = str(searchpath_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_plugins(plugins: Iterable[str]) -> None:
    """Import each declared plugin module so its ``@register`` decorators fire.

    Idempotent (Python caches imports) and tolerant of an already-imported module. ``envs.common``
    is always loaded first: it is dependency-free shared infrastructure (the ``distribution``
    dataset), so every sampling experiment can declare only its true env package.
    """
    _ensure_repo_root_on_path()
    seen: set[str] = set()
    for module in ("envs.common", *plugins):
        if module in seen:
            continue
        seen.add(module)
        importlib.import_module(module)


def load_bundled_envs() -> None:
    """Import every shipped env package (best-effort) so the full catalog registers.

    Used where there is no ``plugins:`` selection — ``forge list`` and the test suite. A missing
    optional env is tolerated (mirrors ``builder.import_builtin_components``)."""
    _ensure_repo_root_on_path()
    for module in BUNDLED_ENVS:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            continue


def load_entrypoint_plugins() -> None:
    """Discover and import every installed package that advertises a ``forge.plugins`` entry point.

    This is the standard Python plugin mechanism (as used by pytest/flake8): forge doesn't scan or
    import arbitrary packages — a package opts in by declaring the entry point, and loading it fires
    the package's ``@register`` decorators. Called from ``import_builtin_components`` so it runs on
    every path (``forge list``, ``train``, ``sample``). A plugin that fails to import is warned and
    skipped: one broken third-party package must not break discovery for the rest."""
    for ep in entry_points(group=FORGE_PLUGIN_GROUP):
        try:
            ep.load()  # importing the referenced module is what fires its @register decorators
        except Exception as exc:  # noqa: BLE001 — isolate one bad plugin from the rest
            warnings.warn(
                f"forge: skipping plugin entry point {ep.name!r}: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )

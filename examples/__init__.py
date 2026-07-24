"""genforge example plugins — the concrete method / sampler / schedule / model / cost / controller /
metric / planning-runner implementations, moved OUT of the installed ``forge`` package so the wheel
stays a lean framework (contracts + wiring + one reference path). They register exactly like any
third-party plugin (``@register`` + the matching ABC); an experiment opts in with ``plugins: [examples]``.

Importing this package (the umbrella) loads the whole catalog — used both by an experiment's
``plugins:`` declaration and by ``forge list`` (via ``load_bundled_examples``). A single missing/broken
category must not hide the rest, so each import is best-effort (mirrors ``load_bundled_envs``).
"""

import importlib

_CATEGORIES = (
    "spaces", "schedules", "models", "methods", "samplers",
    "costs", "control", "metrics", "visualizations", "runners",
)

for _cat in _CATEGORIES:
    try:
        importlib.import_module(f"{__name__}.{_cat}")
    except ModuleNotFoundError:
        continue

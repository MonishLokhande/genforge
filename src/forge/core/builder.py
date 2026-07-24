"""The builder — reads a config and wires components in dependency order, returning a Runner.

Build order:
    space → schedule → criterion → model → method → cost → control → sampler
          → environment → dataset → preprocessor → visualizer → runner

Dependencies are injected at construction (Invariant 4): each component is instantiated with its
configured ``params`` plus any already-built component whose name matches one of its ``__init__``
parameters (e.g. a `Method`'s ``schedule``/``space``; a `Sampler`'s ``model``/``schedule``/
``space``/``control``). No component is threaded per-call.

Adding a component requires no edits here (Invariant 7): it is discovered through the registry.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional

from . import registry
from .resolvers import register_resolvers

# Build order. Earlier categories may be injected into later ones by name.
BUILD_ORDER: tuple[str, ...] = (
    "space",
    "schedule",
    "criterion",
    "model",
    "method",
    "cost",
    "control",
    "sampler",
    "environment",
    "dataset",
    "preprocessor",
    "visualizer",
    "metric",
    "runner",
)

# Built-in component modules to import so their ``@register`` decorators fire. Each phase appends
# the modules it adds. Missing modules are tolerated so the framework imports cleanly mid-build-out.
#
# Concrete envs register via the experiment's ``plugins:`` field (see ``core/plugins.py``) — NOT
# here. Only framework categories (spaces/schedules/models/methods/samplers/costs/control, the
# generic ``standardize``/``minmax`` preprocessors, and runners) are built in.
_BUILTIN_MODULES: tuple[str, ...] = (
    # The framework core's own dependency: the Pin controller used by the base sample loop.
    "forge.core.conditioning",
    # ── the ONE reference implementation of each axis (the runnable 2-D DDPM path) ──
    "forge.spaces.euclidean",
    "forge.schedules.vp",
    "forge.models.mlp",
    "forge.methods.criterion",
    "forge.methods.ddpm",
    "forge.samplers.ddpm",
    "forge.preprocessing.standardize",
    "forge.preprocessing.minmax",
    "forge.runners.training",
    # Generic, env-agnostic renderers kept in-package (scatter = 2-D reference; env_render just
    # dispatches to environment.visualize()). metric_set is the composer, not a concrete metric.
    "forge.visualizations.scatter",
    "forge.visualizations.env_render",
    "forge.metrics.metric_set",
    # Every other method / sampler / schedule / model / cost / controller / concrete metric /
    # planning runner is an out-of-tree EXAMPLE under examples/, loaded via an experiment's
    # `plugins: [examples]` field (see examples/README.md). The package ships contracts + one path.
)


class ConfigurationError(Exception):
    """Raised when a config cannot be built into a runnable pipeline. Fails loudly."""


def import_builtin_components() -> None:
    """Import built-in component modules (and installed entry-point plugins) so registration side
    effects run. Called on every path (`forge list`, train, sample), so third-party plugins
    discovered here are visible everywhere the built-ins are."""
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            # Tolerated: a built-in listed for a future phase may not exist yet.
            continue
    # Third-party plugins installed in the environment, opt-in via the ``forge.plugins`` entry
    # point (standard Python plugin discovery — see core/plugins.load_entrypoint_plugins).
    from .plugins import load_entrypoint_plugins

    load_entrypoint_plugins()
    # Register custom OmegaConf resolvers (add/sub/mul, ...) eagerly so ${mul:}/${sub:}
    # interpolations resolve during plain `compose`, not only inside build(). Idempotent.
    register_resolvers()


def _leaf(cfg: Any, category: str) -> Optional[tuple[str, dict]]:
    """Extract ``(name, params)`` for ``category`` from ``cfg``, or None if absent.

    A category leaf is ``{name: <str>, params: {...}}``. Tolerates OmegaConf and plain dicts.
    """
    node = None
    if hasattr(cfg, "get"):
        node = cfg.get(category, None)
    elif isinstance(cfg, dict):
        node = cfg.get(category)
    if node is None:
        return None
    name = node.get("name") if hasattr(node, "get") else getattr(node, "name", None)
    if not name:
        return None
    raw_params = node.get("params", {}) if hasattr(node, "get") else getattr(node, "params", {})
    params = dict(raw_params) if raw_params else {}
    return str(name), params


def match_init_kwargs(cls: type, pool: dict[str, Any], params: dict) -> dict:
    """Build the kwargs for ``cls``: already-built components whose name matches an ``__init__``
    parameter, overlaid with the explicitly configured ``params`` (config always wins).

    The ONE implementation of dependency-by-name matching (Invariant 4). The builder uses it in the
    build loop; ``MetricSet`` reuses it to build its children — do not hand-copy this logic.

    When the class accepts ``**kwargs`` (forwarding to a parent __init__), also walk the MRO
    so that pool objects matching parent params (e.g. model/method in TrainingRunner) are injected.
    MRO walk only when **kwargs present — keeps the common fast-path cheap.
    """
    sig = inspect.signature(cls.__init__)
    pnames: set[str] = set(sig.parameters)
    # If **kwargs is declared, collect injectable names from ancestor __init__ signatures too.
    if "kwargs" in pnames:
        for parent in cls.__mro__[1:]:
            parent_init = parent.__dict__.get("__init__")
            if parent_init is None:
                continue
            try:
                parent_sig = inspect.signature(parent_init)
            except (ValueError, TypeError):
                continue
            pnames |= set(parent_sig.parameters)
    kwargs: dict[str, Any] = {}
    for pname in pnames:
        if pname in ("self", "args", "kwargs"):
            continue
        if pname in pool:
            kwargs[pname] = pool[pname]
    kwargs.update(params)
    return kwargs


def _available_message() -> str:
    """A human-readable listing of what *is* registered, for actionable errors."""
    reg = registry.registered()
    if not reg:
        return "No components are registered. (Built-in components are added in later phases.)"
    lines = ["Registered components by category:"]
    for category, comps in reg.items():
        lines.append(f"  {category}: {', '.join(comps)}")
    return "\n".join(lines)


def _plugins(cfg: Any) -> list[str]:
    """Extract the experiment's declared plugin modules (concrete env packages), if any."""
    node = cfg.get("plugins", None) if hasattr(cfg, "get") else getattr(cfg, "plugins", None)
    if not node:
        return []
    return [str(p) for p in node]


def build(cfg: Any):
    """Construct all configured components in dependency order and return the ready ``Runner``.

    Raises :class:`ConfigurationError` (listing available options) when the config names no
    components, or when no ``runner`` is configured to return.
    """
    register_resolvers()
    import_builtin_components()

    # Concrete envs are not built in — load the experiment's declared ``plugins:`` so their
    # ``@register`` decorators fire before any ``registry.get`` (Invariant 7). With no declaration
    # (e.g. an old checkpoint's config, or an inline test config relying on auto-registration), fall
    # back to importing every bundled env so the build still resolves.
    from .plugins import load_bundled_envs, load_bundled_examples, load_plugins

    declared = _plugins(cfg)
    if declared:
        load_plugins(declared)
    else:
        load_bundled_envs()
        load_bundled_examples()

    present = [c for c in BUILD_ORDER if _leaf(cfg, c) is not None]
    if not present:
        raise ConfigurationError(
            "Empty or unrecognized config: no components were specified.\n"
            "Expected at least one category leaf of the form "
            "`<category>: {name: <name>, params: {...}}` "
            f"(categories, in build order: {', '.join(BUILD_ORDER)}).\n"
            + _available_message()
        )

    pool: dict[str, Any] = {}
    for category in BUILD_ORDER:
        leaf = _leaf(cfg, category)
        if leaf is None:
            continue
        name, params = leaf
        cls = registry.get(category, name)  # raises a listing KeyError on unknown name
        kwargs = match_init_kwargs(cls, pool, params)
        pool[category] = cls(**kwargs)

    if "runner" not in pool:
        raise ConfigurationError(
            "No `runner` was configured; the builder returns a ready Runner, so one is required.\n"
            + _available_message()
        )
    return pool["runner"]

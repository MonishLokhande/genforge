"""The builder — reads a config and wires components in dependency order, returning a Runner.

Build order:
    space → schedule → model → method → cost → control → sampler
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
    "model",
    "method",
    "cost",
    "control",
    "sampler",
    "environment",
    "dataset",
    "preprocessor",
    "visualizer",
    "runner",
)

# Built-in component modules to import so their ``@register`` decorators fire. Each phase appends
# the modules it adds. Missing modules are tolerated so the framework imports cleanly mid-build-out.
#
# Concrete envs register via the experiment's ``plugins:`` field (see ``core/plugins.py``) — NOT
# here. Only framework categories (spaces/schedules/models/methods/samplers/costs/control, the
# generic ``standardize``/``minmax`` preprocessors, and runners) are built in.
_BUILTIN_MODULES: tuple[str, ...] = (
    # Phase 1 — continuous DDPM on 2-D distributions.
    "forge.spaces.euclidean",
    "forge.schedules.vp",
    "forge.models.mlp",
    "forge.methods.ddpm",
    "forge.samplers.ddpm",
    "forge.runners.training",
    # Phase 1.5 — the preprocessor membrane.
    "forge.preprocessing.standardize",
    "forge.preprocessing.minmax",
    # Phase 2 — flow matching + DDIM (paradigm axis: zero edits to space/schedule base).
    "forge.schedules.flow",
    "forge.methods.flow_matching",
    "forge.methods.ot_cfm",
    "forge.methods.ddpm_huber",
    "forge.samplers.flow",
    "forge.samplers.interpolant",
    "forge.samplers.ddim",
    # Phase 3 — discrete D3PM (space axis: cont/disc confined to space + schedule).
    "forge.spaces.discrete",
    "forge.schedules.discrete",
    "forge.methods.d3pm",
    "forge.methods.mdlm",
    "forge.methods.sedd",
    "forge.samplers.tau_leaping",
    "forge.samplers.sedd",
    "forge.models.categorical",
    "forge.models.transformer",
    # Phase 4 — the control layer (cost + online controllers).
    "forge.costs.halfspace",
    "forge.costs.box",
    "forge.costs.ball",
    "forge.costs.likelihood",
    "forge.control.projection",
    "forge.control.guidance",
    "forge.costs.barrier",
    "forge.control.cbf",
    # Phase 5 — trajectory pipeline (windowed planning).
    "forge.models.temporal_unet",
    "forge.models.temporal_unet_janner",
    "forge.methods.conditional",
    "forge.visualizations.trajectory",
    "forge.runners.planning",
    "forge.runners.policy_training",
    # Phase 6 — amortized control (value / FBSDE), method↔control via checkpoint only.
    "forge.costs.reward",
    "forge.models.value",
    "forge.methods.value_training",
    "forge.methods.fbsde",
    "forge.runners.value_training",
    "forge.control.value_guidance",
    "forge.control.fbsde_control",
)


class ConfigurationError(Exception):
    """Raised when a config cannot be built into a runnable pipeline. Fails loudly."""


def import_builtin_components() -> None:
    """Import built-in component modules so registration side effects run."""
    for module in _BUILTIN_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            # Tolerated: a built-in listed for a future phase may not exist yet.
            continue
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


def _inject(cls: type, pool: dict[str, Any], params: dict) -> dict:
    """Build the kwargs for ``cls``: already-built components whose name matches an ``__init__``
    parameter, overlaid with the explicitly configured ``params`` (config always wins).

    When the class accepts ``**kwargs`` (forwarding to a parent __init__), also walk the MRO
    so that pool objects matching parent params (e.g. model/method in TrainingRunner) are injected.
    MRO walk only when **kwargs present — keeps the common fast-path cheap.
    """
    import inspect as _inspect
    sig = inspect.signature(cls.__init__)
    pnames: set[str] = set(sig.parameters)
    # If **kwargs is declared, collect injectable names from ancestor __init__ signatures too.
    if "kwargs" in pnames:
        for parent in cls.__mro__[1:]:
            parent_init = parent.__dict__.get("__init__")
            if parent_init is None:
                continue
            try:
                parent_sig = _inspect.signature(parent_init)
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
    from .plugins import load_bundled_envs, load_plugins

    declared = _plugins(cfg)
    if declared:
        load_plugins(declared)
    else:
        load_bundled_envs()

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
        kwargs = _inject(cls, pool, params)
        pool[category] = cls(**kwargs)

    if "runner" not in pool:
        raise ConfigurationError(
            "No `runner` was configured; the builder returns a ready Runner, so one is required.\n"
            + _available_message()
        )
    return pool["runner"]

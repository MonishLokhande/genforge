"""The component registry — the single source of truth for component discovery.

Every component joins the framework via ``@register(category, name)`` on a class implementing
the matching contract. The builder reads a Hydra config and asks ``create`` for
each component by ``(category, name)``. Adding a component is one decorated class plus, optionally,
one config leaf — no wiring changes anywhere else (Invariant 7).

Fail loudly: an unknown name raises an error that lists the available options.
"""

from __future__ import annotations

from typing import Callable, TypeVar

# The known component categories, in dependency-build order. The registry does not
# *enforce* this list — registration under an unknown category still works — but `list`/`create`
# present categories in this canonical order so the CLI output is stable and readable.
CATEGORIES: tuple[str, ...] = (
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

# {category: {name: class}}
_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


def register(category: str, name: str) -> Callable[[T], T]:
    """Class decorator: record ``cls`` under ``(category, name)`` and return it unchanged.

    Raises if ``(category, name)`` is already taken — silent shadowing of a component is a
    configuration bug we want to surface immediately.
    """

    def decorator(cls: T) -> T:
        bucket = _REGISTRY.setdefault(category, {})
        if name in bucket:
            existing = bucket[name]
            raise ValueError(
                f"Duplicate registration for ({category!r}, {name!r}): "
                f"{existing.__module__}.{existing.__qualname__} is already registered; "
                f"refusing to overwrite with {cls.__module__}.{cls.__qualname__}."
            )
        bucket[name] = cls
        return cls

    return decorator


def get(category: str, name: str) -> type:
    """Return the registered class for ``(category, name)`` or raise a listing error."""
    bucket = _REGISTRY.get(category)
    if not bucket:
        known = ", ".join(c for c in CATEGORIES if c in _REGISTRY) or "(none registered)"
        raise KeyError(
            f"No components registered under category {category!r}. "
            f"Categories with registered components: {known}."
        )
    if name not in bucket:
        available = ", ".join(sorted(bucket)) or "(none)"
        raise KeyError(
            f"Unknown {category} {name!r}. Available {category}s: {available}."
        )
    return bucket[name]


def create(category: str, name: str, **params):
    """Instantiate the registered ``(category, name)`` with ``**params``.

    On an unknown name, raises an error that lists the available options (fail loudly).
    """
    cls = get(category, name)
    return cls(**params)


def registered() -> dict[str, dict[str, type]]:
    """Return a snapshot of the full registry, ``{category: {name: class}}``.

    Categories are ordered canonically (CATEGORIES first, then any extras). Named ``registered``
    rather than ``list`` to avoid shadowing the builtin; the CLI's ``list`` command renders this.
    """
    ordered: dict[str, dict[str, type]] = {}
    for category in CATEGORIES:
        if category in _REGISTRY:
            ordered[category] = dict(sorted(_REGISTRY[category].items()))
    for category in sorted(_REGISTRY):
        if category not in ordered:
            ordered[category] = dict(sorted(_REGISTRY[category].items()))
    return ordered


def clear() -> None:
    """Empty the registry. For test isolation only."""
    _REGISTRY.clear()

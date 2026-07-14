"""MetricSet — a composite `metric` that runs a LIST of child metrics.

The builder constructs exactly one node per category, so scoring with several metrics needs this
wrapper. Each child is built with the shared ``match_init_kwargs`` dependency-matcher (Invariant 4,
the SAME implementation the builder uses — not a hand-copied filter), so a child receives only the
injected deps its ``__init__`` declares. Child result dicts are merged, **raising on any key
collision** — the flat metrics schema is load-bearing for cross-run aggregation, so a silent
clobber would be a corruption, not a convenience.
"""

from __future__ import annotations

from ..core.builder import match_init_kwargs
from ..core.interfaces import Metric
from ..core.registry import get, register


def _spec_params(spec) -> dict:
    p = spec.get("params", {}) if hasattr(spec, "get") else getattr(spec, "params", {})
    if not p:
        return {}
    from omegaconf import OmegaConf

    return OmegaConf.to_container(p, resolve=True) if OmegaConf.is_config(p) else dict(p)


@register("metric", "metric_set")
class MetricSet(Metric):
    def __init__(self, metrics=None, environment=None, model=None, method=None,
                 dataset=None, schedule=None):
        super().__init__(environment, model, method, dataset, schedule)
        pool = {"environment": environment, "model": model, "method": method,
                "dataset": dataset, "schedule": schedule}
        pool = {k: v for k, v in pool.items() if v is not None}
        self.children = []
        for spec in (metrics or []):
            name = spec["name"] if hasattr(spec, "__getitem__") else spec.name
            cls = get("metric", name)
            self.children.append(cls(**match_init_kwargs(cls, pool, _spec_params(spec))))

    def __call__(self, samples=None, held_out=None) -> dict:
        out: dict = {}
        for child in self.children:
            d = child(samples=samples, held_out=held_out)
            dup = out.keys() & d.keys()
            if dup:
                raise ValueError(
                    f"metric key collision in MetricSet: {type(child).__name__} re-emits "
                    f"{sorted(dup)} (keys must be unique — the flat metrics schema is load-bearing)."
                )
            out.update(d)
        return out

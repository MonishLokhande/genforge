"""Evaluation metrics — swappable `@register("metric", ...)` components (see core.interfaces.Metric).

Sample-driven metrics (distribution/coverage) score generated RAW-unit samples against an
environment reference; data-driven metrics (likelihood) score a NORMALIZED held-out batch through
the model. `MetricSet` runs a list of them.
"""

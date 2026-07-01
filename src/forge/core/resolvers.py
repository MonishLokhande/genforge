"""OmegaConf custom resolvers.

Registered once at startup so configs can reference computed values.
Currently ships arithmetic resolvers (add/sub/mul) for Tier B config expressions such as
``${mul:${n_obs_steps},${obs_dim}}`` and ``${sub:${n_obs_steps},1}``.
"""

from __future__ import annotations

from omegaconf import OmegaConf

_REGISTERED = False


def register_resolvers() -> None:
    """Idempotently register genforge's custom OmegaConf resolvers."""
    global _REGISTERED
    if _REGISTERED:
        return
    OmegaConf.register_new_resolver("add", lambda a, b: a + b, replace=True)
    OmegaConf.register_new_resolver("sub", lambda a, b: a - b, replace=True)
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    _REGISTERED = True

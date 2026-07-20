"""EnvRenderVisualizer — routes rendering to the environment's own ``visualize(samples, out)``.

Env-specific rendering lives *with the env* (a text env writes a decoded transcript, an image env
would write a grid), while similar envs reuse a shared helper — so text diLLM and ARLLM render
identically without the framework knowing anything about text. The builder injects ``environment``
by name (like a metric), so this class stays tiny and env-agnostic.

Best-effort like the other visualizers, but **loud** when the selected env cannot render: selecting
``env_render`` is an explicit request to visualize, so a missing ``visualize()`` must raise, not
silently skip (the design contract fail loudly).
"""

from __future__ import annotations

from pathlib import Path

from ..core.registry import register


@register("visualizer", "env_render")
class EnvRenderVisualizer:
    def __init__(self, environment=None, out_dir: str = "output"):
        self.environment = environment            # injected by the builder
        self.out_dir = out_dir

    def render(self, samples, path: str = "samples.txt"):
        """Hand ``samples`` (raw units, e.g. token ids) to the env's ``visualize(samples, out)``."""
        env = self.environment
        if env is None or not hasattr(env, "visualize"):
            who = type(env).__name__ if env is not None else "no environment"
            raise AttributeError(
                f"visualizer=env_render dispatches to the environment's visualize(samples, out), "
                f"but {who} has none. Add a visualize() to the env, or pick a different visualizer."
            )
        return env.visualize(samples, str(Path(self.out_dir) / path))

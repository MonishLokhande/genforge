"""Minari-format locomotion adapter, registered as ("environment", "minari").

Duck-typed adapter (NO ABC): episodes() yields canonical episode dicts; build_env() recovers the
eval env. All sim/IO imports (minari) are lazy. Friendly D4RL-style aliases register the same class.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from forge.core.registry import register

MINARI_ID_MAP: dict[str, list[str]] = {
    "hopper-medium-v2":             ["mujoco/hopper/medium-v0"],
    "hopper-expert-v2":             ["mujoco/hopper/expert-v0"],
    "hopper-medium-expert-v2":      ["mujoco/hopper/medium-v0", "mujoco/hopper/expert-v0"],
    "walker2d-medium-v2":           ["mujoco/walker2d/medium-v0"],
    "walker2d-expert-v2":           ["mujoco/walker2d/expert-v0"],
    "walker2d-medium-expert-v2":    ["mujoco/walker2d/medium-v0", "mujoco/walker2d/expert-v0"],
    "halfcheetah-medium-v2":        ["mujoco/halfcheetah/medium-v0"],
    "halfcheetah-expert-v2":        ["mujoco/halfcheetah/expert-v0"],
    "halfcheetah-medium-expert-v2": ["mujoco/halfcheetah/medium-v0", "mujoco/halfcheetah/expert-v0"],
}


def resolve_minari_ids(name: str) -> list[str]:
    if name in MINARI_ID_MAP:
        return MINARI_ID_MAP[name]
    if "/" in name:       # raw Minari id passed directly
        return [name]
    raise KeyError(f"Unknown Minari dataset name: {name!r}")


def extract_observations(obs) -> np.ndarray:
    """PointMaze wraps observations in {'observation': ...}; unwrap if needed."""
    if isinstance(obs, dict):
        obs = obs["observation"]
    return np.asarray(obs, dtype=np.float32)


@register("environment", "minari")
class MinariAdapter:
    """Format-pure Minari adapter. No env-specific branches live here."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.distribution = name      # ckpt-key provenance
        try:
            import minari
        except ImportError as exc:
            raise ImportError(
                "Robotics experiments require the robotics dependency group, which is only "
                "installable from a source checkout: git clone the genforge repo and run "
                "`uv sync --group robotics`."
            ) from exc
        self.ids = resolve_minari_ids(name)
        self.datasets = [minari.load_dataset(d, download=True) for d in self.ids]

    def build_env(self):
        return self.datasets[0].recover_environment()

    def episodes(self) -> Iterator[dict]:
        for dataset in self.datasets:
            for ep in dataset.iterate_episodes():
                obs_full = extract_observations(ep.observations)
                actions = np.asarray(ep.actions, dtype=np.float32)
                T = actions.shape[0]
                if T == 0:
                    continue
                yield {
                    "observations":      obs_full[:T],
                    "actions":           actions,
                    "rewards":           np.asarray(ep.rewards, dtype=np.float32)[:T],
                    "terminals":         np.asarray(ep.terminations, dtype=bool)[:T],
                    "timeouts":          np.asarray(ep.truncations, dtype=bool)[:T],
                    "next_observations": obs_full[1: T + 1],
                }


for _name in MINARI_ID_MAP:
    register("environment", _name)(MinariAdapter)

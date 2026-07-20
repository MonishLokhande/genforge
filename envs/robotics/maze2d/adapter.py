"""Maze2D environment adapter, registered as ("environment", "maze2d").

Loads from Minari (D4RL/pointmaze) and yields canonical episode dicts; adds the pointmaze
last-step-truncation fix and goal-proximity terminal tagging. Duck-typed (NO ABC); minari import
is lazy. The goal-tagging (maze2d_set_terminals) is pure numpy.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from forge.core.registry import register

ALIASES: dict[str, str] = {
    "maze2d-large-v1":  "D4RL/pointmaze/large-v2",
    "maze2d-medium-v1": "D4RL/pointmaze/medium-v2",
    "maze2d-umaze-v1":  "D4RL/pointmaze/umaze-v2",
}

DEFAULT = "maze2d-large-v1"


def maze2d_set_terminals(episode: dict, threshold: float = 0.5) -> dict:
    """Flag steps within ``threshold`` world-units of the episode's final (x, y) goal proxy."""
    obs = np.asarray(episode["observations"])
    goal_xy = obs[-1, :2]
    dists = np.linalg.norm(obs[:, :2] - goal_xy, axis=-1)
    return {**episode, "terminals": (dists < float(threshold)).astype(bool)}


@register("environment", "maze2d")
class Maze2DAdapter:
    """Maze2D adapter: loads pointmaze, drops Minari's terminal-overlap step, tags goal terminals."""

    def __init__(self, name: str = DEFAULT, terminal_threshold: float = 0.5):
        self.name = name
        self.distribution = name
        self.terminal_threshold = float(terminal_threshold)
        try:
            import minari
        except ImportError as exc:
            raise ImportError(
                "Robotics experiments require the robotics dependency group, which is only "
                "installable from a source checkout: git clone the genforge repo and run "
                "`uv sync --group robotics`."
            ) from exc
        self.dataset = minari.load_dataset(ALIASES.get(name, name), download=True)

    def build_env(self):
        return self.dataset.recover_environment()

    def episodes(self) -> Iterator[dict]:
        for ep in self.dataset.iterate_episodes():
            obs = ep.observations
            if isinstance(obs, dict):
                obs = obs["observation"]
            obs = np.asarray(obs, dtype=np.float32)
            actions = np.asarray(ep.actions, dtype=np.float32)
            T = actions.shape[0]
            if T < 2:   # trim drops one step; maze2d_set_terminals reads obs[-1]
                continue
            episode = {
                "observations":      obs[:T],
                "actions":           actions,
                "rewards":           np.asarray(ep.rewards, dtype=np.float32)[:T],
                "terminals":         np.asarray(ep.terminations, dtype=bool)[:T],
                "timeouts":          np.asarray(ep.truncations, dtype=bool)[:T],
                "next_observations": obs[1: T + 1],
            }
            episode = {k: v[:-1] for k, v in episode.items()}   # drop Minari's overlap step
            episode = maze2d_set_terminals(episode, self.terminal_threshold)
            yield episode


for _alias in ALIASES:
    register("environment", _alias)(Maze2DAdapter)

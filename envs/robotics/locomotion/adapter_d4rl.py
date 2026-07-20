"""D4RL gym-mujoco flat-HDF5 locomotion adapter, registered as ("environment", "d4rl").

Reads original D4RL v2 buffers directly (no d4rl package). Duck-typed (NO ABC); h5py/gymnasium
imports are lazy. Episode boundaries are transitions where terminals|timeouts is set, plus a
defensive split at max_episode_steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from forge.core.registry import register

D4RL_FILES: dict[str, str] = {
    "hopper-medium-replay-v2":      "hopper_medium_replay-v2.hdf5",
    "walker2d-medium-replay-v2":    "walker2d_medium_replay-v2.hdf5",
    "halfcheetah-medium-replay-v2": "halfcheetah_medium_replay-v2.hdf5",
}

ENV_ID_MAP: dict[str, str] = {
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "halfcheetah": "HalfCheetah-v5",
}


@register("environment", "d4rl")
class D4RLHdf5Adapter:
    """Reads a D4RL flat HDF5 buffer and yields canonical episode dicts."""

    def __init__(self, name: str, *, path: str | None = None,
                 max_episode_steps: int = 1000):
        self.name = name
        self.distribution = name
        self.max_episode_steps = int(max_episode_steps)
        if path is None:
            if name not in D4RL_FILES:
                raise KeyError(
                    f"Unknown D4RL dataset name {name!r}; known: {sorted(D4RL_FILES)} "
                    "(or pass an explicit path=)")
            path = str(Path("data/d4rl") / D4RL_FILES[name])
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"D4RL buffer not found at {self.path}. Fetch all locomotion buffers with "
                f"`bash scripts/download_d4rl.sh`, or download {self.path.name!r} manually from "
                f"http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ into "
                f"{self.path.parent}/ .")

    def build_env(self):
        import gymnasium
        prefix = str(self.distribution).split("-", 1)[0]
        return gymnasium.make(ENV_ID_MAP[prefix])

    def normalized_score(self, raw_return: float) -> float:
        """Diffuser-comparable D4RL score. The hdf5 buffer IS genuine v2 data, so the dataset half
        passes — but ENV_ID_MAP rolls out on v5, so this RAISES until ENV_ID_MAP names the v2 env
        (needs mujoco_py). It returns a real number the moment that env mismatch is resolved."""
        from .normalize import d4rl_normalized_score
        prefix = str(self.distribution).split("-", 1)[0]
        return d4rl_normalized_score(raw_return, name=self.name, dataset_ids=self.name,
                                     env_id=ENV_ID_MAP[prefix])

    def episodes(self) -> Iterator[dict]:
        import h5py
        with h5py.File(self.path, "r") as f:
            obs = np.asarray(f["observations"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = np.asarray(f["rewards"], dtype=np.float32)
            terminals = np.asarray(f["terminals"], dtype=bool)
            timeouts = np.asarray(f["timeouts"], dtype=bool)

        N = actions.shape[0]
        flags = terminals | timeouts
        start = 0
        for i in range(N):
            if not (flags[i] or (i - start + 1) >= self.max_episode_steps or i == N - 1):
                continue
            sl = slice(start, i + 1)
            o = obs[sl]
            yield {
                "observations":      o,
                "actions":           actions[sl],
                "rewards":           rewards[sl],
                "terminals":         terminals[sl],
                "timeouts":          timeouts[sl],
                "next_observations": np.concatenate([o[1:], o[-1:]], axis=0),
            }
            start = i + 1


for _friendly in D4RL_FILES:
    register("environment", _friendly)(D4RLHdf5Adapter)

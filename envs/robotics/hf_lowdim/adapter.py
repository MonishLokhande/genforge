"""Low-dimensional adapter for HuggingFace-hosted (lerobot-format) datasets.

Loads the tabular columns of a lerobot-format dataset via plain HF ``datasets``
(parquet-backed; video columns are NOT decoded — lowdim only), splits episodes on
the ``episode_index`` column, and builds the matching gymnasium env (gym-pusht /
gym-aloha) for rollout. The ``lerobot`` package is not used or required — only
``datasets`` + the gym env package.

Named ``hf_lowdim`` (registered as ``("environment", "hf_lowdim")``) because it is
a generic loader: the ``lerobot`` package is neither used nor required.

All imports of ``datasets``, ``gymnasium``, ``gym_pusht``, ``gym_aloha`` are lazy
(inside methods) — this module imports cleanly without any of them installed.

PushT keypoints (the diffusion_policy lowdim task): use
``name=lerobot/pusht_keypoints`` with
``obs_keys=[observation.environment_state, observation.state]`` — 16-d T-block
keypoints ⊕ 2-d agent position = 18-d obs. ``build_env()`` then creates
``gym_pusht/PushT-v0`` with ``obs_type="environment_state_agent_pos"``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Iterator, Sequence

import numpy as np

from forge.core.registry import register

# HF repo_id → (gym env id, gym.make kwargs). The PushT kwargs select the Dict
# obs space matching the keypoints dataset; plain gym.make defaults otherwise.
LEROBOT_ENV_SPECS: dict[str, tuple[str, dict]] = {
    "lerobot/pusht": ("gym_pusht/PushT-v0", {"obs_type": "environment_state_agent_pos"}),
    "lerobot/pusht_keypoints": ("gym_pusht/PushT-v0", {"obs_type": "environment_state_agent_pos"}),
    "lerobot/pusht_image": ("gym_pusht/PushT-v0", {"obs_type": "environment_state_agent_pos"}),
    "lerobot/aloha_sim_insertion_human": ("gym_aloha/AlohaInsertion-v0", {}),
    "lerobot/aloha_sim_insertion_scripted": ("gym_aloha/AlohaInsertion-v0", {}),
    "lerobot/aloha_sim_transfer_cube_human": ("gym_aloha/AlohaTransferCube-v0", {}),
    "lerobot/aloha_sim_transfer_cube_scripted": ("gym_aloha/AlohaTransferCube-v0", {}),
}

# Dataset column → key in the env's Dict observation space (gym-pusht naming).
OBS_KEY_TO_ENV_KEY: dict[str, str] = {
    "observation.environment_state": "environment_state",
    "observation.state": "agent_pos",
}


@register("environment", "hf_lowdim")
class HFLowdimAdapter:
    """Adapter for low-dimensional HuggingFace-backed (lerobot-format) datasets.

    Args:
        name: HF repo id (doubles as ``repo_id`` when that is not given).
        repo_id: explicit HF repo id, e.g. ``lerobot/pusht_keypoints``.
        obs_keys: dataset columns concatenated (in order) into the observation
            vector. The same order drives :meth:`flatten_env_obs` at rollout.
        env_id: gym env id; resolved from :data:`LEROBOT_ENV_SPECS` when None.
        env_kwargs: extra ``gym.make`` kwargs, merged over the spec defaults.
        dataset: pre-loaded HF dataset (dependency injection for tests /
            local data); skips the hub download when given.
    """

    def __init__(
        self,
        name: str = "lerobot/pusht_keypoints",
        *,
        repo_id: str | None = None,
        obs_keys: Sequence[str] = ("observation.state",),
        env_id: str | None = None,
        env_kwargs: dict | None = None,
        dataset=None,
        **kwargs,
    ):
        self.name = name
        self.repo_id = repo_id or name
        self.distribution = self.repo_id.split("/")[-1]  # runner.ckpt_key() reads this
        self.obs_keys = list(obs_keys)
        spec_env_id, spec_kwargs = LEROBOT_ENV_SPECS.get(self.repo_id, (None, {}))
        self.env_id = env_id or spec_env_id
        self.env_kwargs = {**spec_kwargs, **(env_kwargs or {})}
        self.dataset = dataset if dataset is not None else self.load_hf_dataset()

    def load_hf_dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "`datasets` is required for HFLowdimAdapter. "
                "Install it with: uv sync --group robotics"
            ) from e
        return load_dataset(self.repo_id, split="train")

    def episode_bounds(self) -> list[tuple[int, int]]:
        ep_col = np.asarray(self.dataset["episode_index"])
        _unused, starts = np.unique(ep_col, return_index=True)
        ends = [*map(int, starts[1:]), len(ep_col)]
        return [(int(start), end) for start, end in zip(starts, ends)]

    def column(self, hf_slice: dict, key: str) -> np.ndarray:
        if key not in hf_slice:
            raise KeyError(
                f"Column '{key}' not in dataset. "
                f"Available: {list(hf_slice.keys())}"
            )
        return np.asarray(hf_slice[key], dtype=np.float32)

    def ensure_env_pkg(self) -> None:
        """Validate that the gym env id and its package are available."""
        if self.env_id is None:
            raise RuntimeError(
                f"No gym env registered for repo '{self.repo_id}'. "
                "Pass `env_id=...` explicitly."
            )
        try:
            if self.env_id.startswith("gym_pusht"):
                import_module("gym_pusht")
            elif self.env_id.startswith("gym_aloha"):
                import_module("gym_aloha")
        except ImportError as e:
            raise ImportError(
                f"The env package for {self.env_id} is not installed. "
                "Install gym-pusht / gym-aloha as appropriate."
            ) from e

    def build_env(self):
        self.ensure_env_pkg()
        import gymnasium as gym
        return gym.make(self.env_id, **self.env_kwargs)

    def flatten_env_obs(self, obs) -> np.ndarray:
        """Flatten a Dict env observation into the dataset's obs layout.

        Keys are taken in ``obs_keys`` order (mapped through
        :data:`OBS_KEY_TO_ENV_KEY`), so the rollout vector matches the columns
        the model was trained on. Array observations pass through unchanged.
        """
        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float32).reshape(-1)
        parts = []
        for key in self.obs_keys:
            env_key = OBS_KEY_TO_ENV_KEY.get(key, key)
            if env_key not in obs:
                raise KeyError(
                    f"Env observation missing '{env_key}' (mapped from '{key}'). "
                    f"Available: {list(obs.keys())}"
                )
            parts.append(np.asarray(obs[env_key], dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)

    def episodes(self) -> Iterator[dict]:
        for start, end in self.episode_bounds():
            T = end - start
            if T < 2:
                continue

            hf_slice = self.dataset[start:end]

            observations = np.concatenate(
                [self.column(hf_slice, key).reshape(T, -1) for key in self.obs_keys],
                axis=1,
            )
            actions = self.column(hf_slice, "action").reshape(T, -1)

            rewards = (
                self.column(hf_slice, "next.reward").reshape(T)
                if "next.reward" in hf_slice
                else np.zeros(T, dtype=np.float32)
            )

            if "next.done" in hf_slice:
                terminals = self.column(hf_slice, "next.done").reshape(T).astype(bool)
            else:
                terminals = np.zeros(T, dtype=bool)
                terminals[-1] = True

            timeouts = np.zeros(T, dtype=bool)
            timeouts[-1] = not terminals[-1]

            yield {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminals": terminals,
                "timeouts": timeouts,
                "next_observations": np.concatenate(
                    [observations[1:], observations[-1:]], axis=0
                ),
            }


__all__ = ["HFLowdimAdapter", "LEROBOT_ENV_SPECS", "OBS_KEY_TO_ENV_KEY"]

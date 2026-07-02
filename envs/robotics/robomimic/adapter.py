"""Robomimic low-dimensional HDF5 adapter (lowdim only; no image/render/replay).

Iterates ``data/demo_N`` groups of a robomimic (v0.4) HDF5 demonstration file,
flattens the configured low-dim observation keys, and builds the simulation env
for rollout from the ``env_args`` metadata embedded in the file.

Default observation keys follow diffusion_policy's order
``(object, eef_pos, eef_quat, gripper)`` — lift task: 10+3+4+2 = 19-d obs.
Rollout evaluation runs serially (robomimic ``EnvBase`` does not support
parallel instantiation); its 4-tuple ``step`` API is handled by
``MultiStepWrapper``.

All robosuite/robomimic imports are lazy (inside methods) — no sim needed to
iterate episodes or run the test suite.
"""
from __future__ import annotations

import json
from typing import Iterator, Sequence

import numpy as np

from forge.core.registry import register

# diffusion_policy task-config order (task/*_lowdim.yaml).
DEFAULT_LOW_DIM_KEYS: tuple[str, ...] = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

# A direct robosuite env names the object block "object-state"; robomimic's
# processed datasets name it "object". Alias so flatten_env_obs serves both the
# robomimic-wrapper rollout and the direct-robosuite render rollout.
OBS_KEY_ALIASES: dict[str, str] = {"object": "object-state"}


@register("environment", "robomimic")
class RobomimicAdapter:
    """Adapter for robomimic low-dimensional HDF5 datasets.

    Args:
        name: label for checkpoint namespacing; doubles as the file path when
            ``path`` is not given.
        path: HDF5 demonstration file (e.g. ``data/robomimic/lift/ph/low_dim_v15.hdf5``).
        obs_keys: low-dim observation keys, flattened in this order (defaults
            to dp's ``(object, eef_pos, eef_quat, gripper)`` subset present in
            the file).
        filter_key: optional ``mask/<filter_key>`` demo subset (train/valid).
    """

    def __init__(
        self,
        name: str = "robomimic",
        *,
        path: str | None = None,
        obs_keys: Sequence[str] | None = None,
        filter_key: str | None = None,
        **kwargs,
    ):
        self.name = name
        self.path = path or name
        self.distribution = name      # runner.ckpt_key() reads this
        self.obs_keys = list(obs_keys) if obs_keys is not None else None
        self.filter_key = filter_key
        self.env_meta: dict | None = None

    def resolve_obs_keys(self, obs_grp) -> list[str]:
        available = list(obs_grp.keys())

        if self.obs_keys is not None:
            missing = [k for k in self.obs_keys if k not in available]
            if missing:
                raise KeyError(
                    f"Requested obs keys not in dataset: {missing}. "
                    f"Available keys: {available}"
                )
            return list(self.obs_keys)

        keys = [k for k in DEFAULT_LOW_DIM_KEYS if k in available]
        if not keys:
            raise ValueError(
                f"No default low-dim keys found in {self.path}. "
                f"Pass `obs_keys=` explicitly. Available: {available}"
            )
        return keys

    @staticmethod
    def flatten_obs(obs_grp, keys: Sequence[str]) -> np.ndarray:
        # Read raw ([()]) then cast in numpy — HDF5 lacks conversion paths for
        # some dtype pairs (e.g. float64→bool) when converting during the read.
        return np.concatenate(
            [
                np.asarray(obs_grp[k][()]).astype(np.float32).reshape(len(obs_grp[k]), -1)
                for k in keys
            ],
            axis=1,
        )

    def flatten_env_obs(self, obs) -> np.ndarray:
        """Flatten a robomimic dict observation in ``obs_keys`` order so the
        rollout vector matches the columns the model was trained on."""
        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float32).reshape(-1)

        def resolve(key):
            if key in obs:
                return key
            alias = OBS_KEY_ALIASES.get(key)
            return alias if alias in obs else None

        keys = self.obs_keys
        if keys is None:
            keys = [k for k in DEFAULT_LOW_DIM_KEYS if resolve(k) is not None]
        parts = []
        for key in keys:
            src = resolve(key)
            if src is None:
                raise KeyError(
                    f"Env observation missing '{key}'. Available: {list(obs.keys())}"
                )
            parts.append(np.asarray(obs[src], dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)

    def demo_names(self, f) -> list[str]:
        if self.filter_key is None:
            names = list(f["data"].keys())
        else:
            names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f[f"mask/{self.filter_key}"][:]
            ]
        return sorted(names, key=lambda s: int(s.split("_")[-1]))

    def read_env_meta(self) -> dict:
        if self.env_meta is None:
            import h5py
            with h5py.File(self.path, "r") as f:
                env_args = f["data"].attrs["env_args"]
                if isinstance(env_args, bytes):
                    env_args = env_args.decode()
                self.env_meta = json.loads(env_args)
        return self.env_meta

    def ensure_obs_utils(self) -> None:
        """Initialize robomimic's global obs-modality registry if it is unset.

        ``EnvRobosuite.get_observation()`` indexes the module global
        ``ObsUtils.OBS_KEYS_TO_MODALITIES``, which stays ``None`` until a
        robomimic training/eval pipeline calls one of its ``initialize_obs_*``
        helpers. Building an env standalone (as we do) skips that, so the first
        ``reset()`` crashes with ``argument of type 'NoneType' is not iterable``
        (env_robosuite.py: ``k in None``). Registering the low-dim obs keys here
        is the missing initialization — not a robosuite-version workaround. Only
        runs when unset, so it never clobbers a real robomimic training setup."""
        import robomimic.utils.obs_utils as ObsUtils
        if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
            keys = list(self.obs_keys) if self.obs_keys is not None else list(DEFAULT_LOW_DIM_KEYS)
            ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": keys})

    def build_env(self):
        try:
            import robomimic.utils.env_utils as EnvUtils
        except ImportError as e:
            raise ImportError(
                "Robotics experiments require the robotics dependency group, which is only "
                "installable from a source checkout: git clone the genforge repo and run "
                "`uv sync --group robotics`."
            ) from e
        self.ensure_obs_utils()
        return EnvUtils.create_env_from_metadata(
            env_meta=self.read_env_meta(),
            render=False,
        )

    @staticmethod
    def env_success(env, reward: float) -> bool:
        """Best-effort task-success read across robomimic env API variants."""
        check = getattr(env, "is_success", None)
        if callable(check):
            out = check()
            if isinstance(out, dict):
                return bool(out.get("task", False))
            return bool(out)
        return reward > 0.0

    def episodes(self) -> Iterator[dict]:
        import h5py
        with h5py.File(self.path, "r") as f:
            data = f["data"]

            for demo_name in self.demo_names(f):
                demo = data[demo_name]
                obs_keys = self.resolve_obs_keys(demo["obs"])

                observations = self.flatten_obs(demo["obs"], obs_keys)
                actions = np.asarray(demo["actions"][()]).astype(np.float32)
                T = actions.shape[0]

                if T == 0:
                    continue

                rewards = (
                    np.asarray(demo["rewards"][()]).astype(np.float32)
                    if "rewards" in demo
                    else np.zeros(T, dtype=np.float32)
                )
                terminals = (
                    np.asarray(demo["dones"][()]).astype(bool)
                    if "dones" in demo
                    else np.zeros(T, dtype=bool)
                )[:T]

                timeouts = np.zeros(T, dtype=bool)
                timeouts[-1] = not terminals[-1]

                next_observations = (
                    self.flatten_obs(demo["next_obs"], obs_keys)
                    if "next_obs" in demo
                    else np.concatenate([observations[1:], observations[-1:]], axis=0)
                )

                yield {
                    "observations": observations[:T],
                    "actions": actions,
                    "rewards": rewards[:T],
                    "terminals": terminals,
                    "timeouts": timeouts,
                    "next_observations": next_observations[:T],
                }


__all__ = ["RobomimicAdapter", "DEFAULT_LOW_DIM_KEYS"]

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

# Camera order for vision policies — frozen to the order the trained image checkpoints declare.
# NOTE a vision policy's `obs_keys` is NOT DEFAULT_LOW_DIM_KEYS: it drops the privileged "object"
# state (the camera is meant to supply it), giving proprio_dim=9 rather than 23. An image
# experiment must therefore set `obs_keys` explicitly.
DEFAULT_IMAGE_KEYS: tuple[str, ...] = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)


class LazyEpisodeImages:
    """One episode's camera frames, read from the HDF5 on demand.

    Yielded by ``episodes()`` instead of a materialized array so the DATASET decides the policy:
    ``np.asarray(...)`` (via ``__array__``) pulls the whole episode for the in-RAM path, while
    ``read(local)`` fetches just the frames a batch needs for the streaming path. One flag
    (``dataset.stream``) therefore controls RAM-vs-stream, and the eager path is byte-identical
    to a direct read.

    Two HDF5 landmines this exists to handle:
      * handles are NOT fork-safe — a handle opened in the parent and touched in a DataLoader
        worker misbehaves, so it is (re)opened per worker and dropped from the pickled state.
      * fancy indexing requires STRICTLY INCREASING, duplicate-free indices — `read` sorts and
        de-duplicates, and the caller restores the original order.
    """

    def __init__(self, path: str, demo: str, keys: Sequence[str], length: int, frame_shape):
        self.path, self.demo, self.keys = str(path), str(demo), list(keys)
        self.length, self.frame_shape = int(length), tuple(frame_shape)   # frame: (C, H, W)
        self._f = None
        self._wid = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_f"] = d["_wid"] = None      # h5py handles cannot be pickled to a worker
        return d

    def _obs_grp(self):
        import h5py
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        wid = -1 if info is None else info.id
        if self._f is None or self._wid != wid:
            self._f = h5py.File(self.path, "r")     # per-worker handle (fork-safety)
            self._wid = wid
        return self._f[f"data/{self.demo}/obs"]

    def __len__(self) -> int:
        return self.length

    @property
    def shape(self):
        return (self.length, len(self.keys), *self.frame_shape)

    def read(self, local) -> np.ndarray:
        """Frames at ``local`` (any order, duplicates allowed) → ``(n, n_cam, C, H, W)`` uint8."""
        local = np.asarray(local).reshape(-1)
        uniq, inv = np.unique(local, return_inverse=True)   # sorted + de-duped for h5py
        grp = self._obs_grp()
        frames = np.stack(
            [np.asarray(grp[k][uniq]).transpose(0, 3, 1, 2) for k in self.keys], axis=1
        )
        return frames[inv]                                   # restore the caller's order

    def __array__(self, dtype=None):
        out = self.read(np.arange(self.length))
        return out if dtype is None else out.astype(dtype)


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
        image_keys: Sequence[str] | None = None,
        filter_key: str | None = None,
    ):
        self.name = name
        self.path = path or name
        self.distribution = name      # runner.ckpt_key() reads this
        self.obs_keys = list(obs_keys) if obs_keys is not None else None
        # None => lowdim adapter, byte-identical to before. Set it to opt into camera frames:
        # episodes() then yields ep["images"] and stack_env_images() serves the rollout.
        self.image_keys = list(image_keys) if image_keys is not None else None
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

    @staticmethod
    def stack_images(obs_grp, keys: Sequence[str]) -> np.ndarray:
        """HDF5 camera frames → ``(T, n_cam, C, H, W)`` uint8, in ``keys`` order.

        The image datasets already store DECODED, RESIZED ``(T, 84, 84, 3)`` uint8 per camera, so
        this is a key read + an HWC→CHW transpose. No decode, no resize, no PIL/cv2.
        uint8 is preserved end-to-end — the vision encoder does the ``/255`` (Inv 9).
        """
        return np.stack(
            [np.asarray(obs_grp[k][()]).transpose(0, 3, 1, 2) for k in keys], axis=1
        )

    def stack_env_images(self, obs) -> np.ndarray:
        """Env dict observation → ``(n_cam, C, H, W)`` uint8, in ``image_keys`` order.

        Paired with `stack_images` so ONE object owns camera order + layout on both the HDF5 and
        the rollout side — that pairing is what keeps train and rollout from skewing. robomimic's
        ``rgb`` modality hands back the SAME ``(84, 84, 3)`` uint8 HWC the HDF5 stores, so this is
        the identical transpose (verified: env vs HDF5 frames agree to mean |diff| < 1/255).
        """
        keys = self.image_keys or list(DEFAULT_IMAGE_KEYS)
        parts = []
        for key in keys:
            if key not in obs:
                raise KeyError(
                    f"Env observation missing image key '{key}'. Available: {list(obs.keys())}. "
                    "Was the env built with use_image_obs=True and the rgb modality registered?"
                )
            frame = np.asarray(obs[key])
            if frame.ndim == 3 and frame.shape[-1] in (1, 3):   # HWC → CHW
                frame = frame.transpose(2, 0, 1)
            parts.append(frame)
        return np.stack(parts, axis=0)

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
        """Register THIS adapter's obs modalities in robomimic's global registry.

        ``EnvRobosuite.get_observation()`` indexes the module global
        ``ObsUtils.OBS_KEYS_TO_MODALITIES``, which stays ``None`` until a
        robomimic training/eval pipeline calls one of its ``initialize_obs_*``
        helpers. Building an env standalone (as we do) skips that, so the first
        ``reset()`` crashes with ``argument of type 'NoneType' is not iterable``
        (env_robosuite.py: ``k in None``). Registering the obs keys here is the
        missing initialization — not a robosuite-version workaround.

        Keyed on whether the global already maps OUR keys, NOT on whether it is
        unset: the registry is process-global, so the first adapter to build an
        env used to win it for the whole process. A lowdim run followed by an
        image run in one process (any benchmark sweep does exactly this) left
        ``rgb`` unregistered, and the cameras were then dropped from the env
        observation. Re-initializing is a no-op whenever the mapping already
        satisfies us, so an equivalent setup is still never clobbered."""
        import robomimic.utils.obs_utils as ObsUtils
        keys = list(self.obs_keys) if self.obs_keys is not None else list(DEFAULT_LOW_DIM_KEYS)
        mapping: dict[str, list[str]] = {"low_dim": keys}
        if self.image_keys is not None:
            # Without an `rgb` entry EnvRobosuite.get_observation() mishandles the camera keys.
            mapping["rgb"] = list(self.image_keys)
        current = ObsUtils.OBS_KEYS_TO_MODALITIES or {}     # key -> modality
        if any(current.get(k) != modality for modality, ks in mapping.items() for k in ks):
            # MERGE, don't clobber: initialize_* REPLACES the process-global map, so registering
            # only our keys would drop a live sibling adapter's modalities (rgb especially → the
            # other env silently loses its cameras). Fold our keys into the existing union; ours
            # wins on a per-key modality conflict.
            combined = {**current, **{k: mod for mod, ks in mapping.items() for k in ks}}
            union: dict[str, list[str]] = {}
            for k, mod in combined.items():
                union.setdefault(mod, []).append(k)
            ObsUtils.initialize_obs_modality_mapping_from_dict(union)

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
        images = self.image_keys is not None
        # Offscreen rendering is what actually produces the camera frames; the dataset's own
        # env_args already declare the cameras + 84x84, robomimic just has to be told to keep them.
        # Headless needs MUJOCO_GL=egl (or osmesa) or mujoco raises a FatalError on context init.
        return EnvUtils.create_env_from_metadata(
            env_meta=self.read_env_meta(),
            render=False,
            render_offscreen=images,
            use_image_obs=images,
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

                episode = {
                    "observations": observations[:T],
                    "actions": actions,
                    "rewards": rewards[:T],
                    "terminals": terminals,
                    "timeouts": timeouts,
                    "next_observations": next_observations[:T],
                }
                if self.image_keys is not None:
                    # A lazy accessor, NOT an array: the dataset materializes it (in-RAM path) or
                    # reads per batch (streaming). uint8 (T, n_cam, C, H, W) either way (Inv 9).
                    h, w, _c = demo["obs"][self.image_keys[0]].shape[1:]
                    episode["images"] = LazyEpisodeImages(
                        self.path, demo_name, self.image_keys, T, frame_shape=(3, h, w)
                    )
                yield episode


__all__ = ["RobomimicAdapter", "DEFAULT_LOW_DIM_KEYS", "DEFAULT_IMAGE_KEYS"]

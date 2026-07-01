"""Receding-horizon env wrapper (diffusion_policy MultiStepWrapper).

Stacks the last ``n_obs_steps`` observations and consumes ``n_action_steps``-long action
chunks per :meth:`step`. Warm-up padding replicates the earliest available obs (matches the
dataset's ``pad_before`` edge replication) — any other convention (zero-pad) is a silent
train/eval obs-distribution mismatch at episode boundaries. Rewards aggregate over the whole
episode (``max`` by default): PushT max coverage, robomimic sparse-success indicator.

lowdim only; vision (image_transform/render) dropped from the original —
add the image deque + dict obs back when an image policy lands.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np


def _is_sim_divergence(exc: BaseException) -> bool:
    """True for simulator-integration blowups (dm_control ``PhysicsError`` / MuJoCo). A bad/untrained
    policy can drive the sim to an invalid state (e.g. ``mjWARN_BADQACC``); one such episode should
    end gracefully rather than crash the whole rollout. Narrow by class name / module so genforge
    bugs are NOT swallowed."""
    if type(exc).__name__ == "PhysicsError":
        return True
    module = type(exc).__module__ or ""
    return module.startswith("dm_control") or module.startswith("mujoco")


def aggregate(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    if how == "max":
        return float(np.max(values))
    if how == "min":
        return float(np.min(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "sum":
        return float(np.sum(values))
    raise ValueError(f"unknown reward aggregation {how!r}")


class MultiStepWrapper:
    """Obs-stacking, chunk-consuming wrapper around a single env (lowdim)."""

    def __init__(
        self,
        env: Any,
        n_obs_steps: int,
        n_action_steps: int,
        max_episode_steps: int | None = None,
        reward_agg: str = "max",
        obs_transform: Callable[[Any], np.ndarray] | None = None,
    ) -> None:
        if n_obs_steps <= 0 or n_action_steps <= 0:
            raise ValueError(
                f"n_obs_steps/n_action_steps must be > 0, got {n_obs_steps}/{n_action_steps}"
            )
        self.env = env
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.max_episode_steps = max_episode_steps
        self.reward_agg = reward_agg
        self.obs_transform = obs_transform
        self.obs: deque = deque(maxlen=self.n_obs_steps)
        self.rewards: list[float] = []
        self.steps_taken = 0
        self.done = False

    def transform(self, raw_obs: Any) -> np.ndarray:
        if self.obs_transform is not None:
            return np.asarray(self.obs_transform(raw_obs), dtype=np.float32)
        return np.asarray(raw_obs, dtype=np.float32)

    def stack_deque(self, dq: deque) -> np.ndarray:
        """Front-fill the last ``n_obs_steps`` entries by repeating the earliest one
        (edge replication, dp ``stack_last_n_obs``)."""
        avail = list(dq)
        pad = self.n_obs_steps - len(avail)
        return np.stack([avail[0]] * pad + avail, axis=0)

    def stacked_obs(self) -> np.ndarray:
        return self.stack_deque(self.obs)

    def record_obs(self, raw_obs) -> None:
        self.obs.append(self.transform(raw_obs))

    def reset(self, seed: int | None = None) -> np.ndarray:
        try:
            out = self.env.reset(seed=seed)
        except TypeError:   # robomimic EnvBase: reset() takes no seed
            out = self.env.reset()
        raw_obs = out[0] if isinstance(out, tuple) else out
        self.obs = deque(maxlen=self.n_obs_steps)
        self.record_obs(raw_obs)
        self.rewards = []
        self.steps_taken = 0
        self.done = False
        return self.stacked_obs()

    def step_env_once(self, action) -> tuple[Any, float, bool, dict]:
        out = self.env.step(action)
        if len(out) == 5:   # gymnasium 5-tuple
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated) or bool(truncated), info
        obs, reward, done, info = out   # robomimic EnvBase 4-tuple
        return obs, float(reward), bool(done), info

    def step(self, action_chunk) -> tuple[np.ndarray, float, bool, dict]:
        """Execute up to ``n_action_steps`` actions; return the new obs stack, the running
        episode reward aggregate, the done flag, and the last info."""
        info: dict = {}
        for action in np.asarray(action_chunk)[: self.n_action_steps]:
            if self.done:
                break
            try:
                raw_obs, reward, done, info = self.step_env_once(action)
            except Exception as exc:                    # narrow — only sim divergence
                if not _is_sim_divergence(exc):
                    raise
                self.done = True                        # diverged sim → terminate this episode
                info = {"sim_error": True}
                break
            self.record_obs(raw_obs)
            self.rewards.append(reward)
            self.steps_taken += 1
            if (self.max_episode_steps is not None
                    and self.steps_taken >= self.max_episode_steps):
                done = True
            self.done = done
        return self.stacked_obs(), aggregate(self.rewards, self.reward_agg), self.done, info

    def episode_score(self) -> float:
        """Aggregated episode reward (the rollout metric)."""
        return aggregate(self.rewards, self.reward_agg)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


class PolicyWrapper:
    """Turns a (sampler, preprocessor) pair into ``predict_action`` (receding-horizon).

    Each control cycle the sampler generates a full ``(B, H, Da)`` trajectory from noise,
    conditioned on the observation history via a flat ``(B, To*Do)`` tensor passed as ``cond``.
    Only the ``n_action_steps`` actions starting at ``n_obs_steps-1`` are executed before
    re-planning.

    The obs history reaches the model through ``cond`` (flat, un-normalized) and is NEVER pushed
    through the membrane (Inv 9 — obs normalization is the model's ObsNormalizer). The
    preprocessor's ``inverse`` is applied only to the generated actions. ``ema`` (a genforge
    ``EMA``) samples under the averaged weights via store/copy_to/restore, then restores the live
    weights.

    lowdim only — no vision (obs_images) / amp; add when an image policy lands.
    """

    def __init__(
        self,
        sampler,
        preprocessor=None,
        *,
        n_obs_steps: int,
        n_action_steps: int,
        sample_shape: tuple,
        n_sample_steps: int = 100,
        ema=None,
    ) -> None:
        horizon = sample_shape[0]
        if n_obs_steps - 1 + n_action_steps > horizon:
            raise ValueError(
                f"n_obs_steps-1 + n_action_steps = {n_obs_steps - 1 + n_action_steps} "
                f"exceeds the sampler horizon {horizon}"
            )
        self.sampler = sampler
        self.preprocessor = preprocessor
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.sample_shape = tuple(sample_shape)
        self.n_sample_steps = int(n_sample_steps)
        self.ema = ema

    def device(self):
        return next(self.sampler.model.parameters()).device

    def predict_action(self, obs_stack):
        """Plan one action chunk from an observation history ``(To, Do)`` / ``(B, To, Do)``.
        Returns ``(Ta, Da)`` (or batched) in raw action space."""
        import torch
        obs = torch.as_tensor(np.asarray(obs_stack), dtype=torch.float32)
        unbatched = obs.ndim == 2
        if unbatched:
            obs = obs.unsqueeze(0)                          # (1, To, Do)
        if obs.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"obs stack has {obs.shape[1]} steps, expected n_obs_steps={self.n_obs_steps}"
            )
        obs = obs.to(self.device())
        # Inv 9: flatten to (B, To*Do) — raw, un-normalized; model's ObsNormalizer handles it
        cond = obs.reshape(obs.shape[0], -1)               # (B, To*Do)

        model = self.sampler.model
        model.eval()
        shape = (obs.shape[0], *self.sample_shape)
        if self.ema is not None:
            self.ema.store(model)
            self.ema.copy_to(model)
        try:
            with torch.no_grad():
                result = self.sampler.sample(shape, self.n_sample_steps, cond=cond)
        finally:
            if self.ema is not None:
                self.ema.restore(model)

        actions = result.samples                            # (B, H, Da) normalized
        if self.preprocessor is not None:
            actions = self.preprocessor.inverse(actions)
        start = self.n_obs_steps - 1                        # oa_step_convention
        chunk = actions[:, start : start + self.n_action_steps]
        return chunk[0] if unbatched else chunk


__all__ = ["MultiStepWrapper", "PolicyWrapper", "aggregate"]

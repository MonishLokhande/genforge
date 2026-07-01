"""Policy training runner: TrainingRunner + environment-rollout evaluation.

Training is inherited from ``TrainingRunner`` unchanged; ``evaluate(step)`` is replaced by a
receding-horizon rollout in the adapter's env, returning per-episode scores (PushT max coverage,
robomimic sparse success — both fall out of ``MultiStepWrapper``'s max aggregation). With the
default ``n_eval_envs=1`` episodes run serially over one env; ``n_eval_envs>1`` runs them in waves.

lowdim only, no vision; no validate_wiring (the conditional-model To/cond_dim coupling
isn't exposed yet); summarize inlined — swap to the shared eval-metrics util when it's ported.
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.registry import register
from .multistep import MultiStepWrapper, PolicyWrapper
from .training import TrainingRunner


def _mean_score(scores: list[float], success_threshold: float | None = None) -> dict[str, float]:
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan")}
    out = {"n": int(arr.size), "mean": float(arr.mean()),
           "std": float(arr.std()), "max": float(arr.max()), "min": float(arr.min())}
    if success_threshold is not None:
        out["success_rate"] = float((arr >= success_threshold).mean())
    return out


@register("runner", "policy_training")
class PolicyTrainingRunner(TrainingRunner):
    """TrainingRunner with diffusion_policy rollout evaluation.

    Args:
        n_obs_steps: observation history length To.
        n_action_steps: executed action-chunk length Ta.
        n_rollout_episodes: episodes per evaluation (<=0 disables eval).
        max_episode_steps: per-episode env-step budget.
        rollout_seed: base env seed; episode ``i`` resets with ``seed + i``.
        n_eval_envs: >1 runs episodes in waves over that many cached envs (batched sampling).
        success_threshold: when set, also report ``success_rate`` (PushT coverage 0.95).
    """

    def __init__(
        self, *,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        n_rollout_episodes: int = 10,
        max_episode_steps: int = 300,
        rollout_seed: int = 100000,
        n_eval_envs: int = 1,
        success_threshold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.n_rollout_episodes = int(n_rollout_episodes)
        self.max_episode_steps = int(max_episode_steps)
        self.rollout_seed = int(rollout_seed)
        self.n_eval_envs = int(n_eval_envs)
        self.success_threshold = (None if success_threshold is None
                                  else float(success_threshold))
        self.eval_wrappers: list[MultiStepWrapper] | None = None

    # ── rollout machinery ─────────────────────────────────────────────────────
    def make_policy(self) -> PolicyWrapper:
        return PolicyWrapper(
            self.sampler, self.preprocessor,
            n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps,
            sample_shape=self.dataset.sample_shape, n_sample_steps=self.n_sample_steps,
            ema=self.ema,
        )

    def make_rollout_wrapper(self) -> MultiStepWrapper:
        obs_transform = getattr(self.environment, "flatten_env_obs", None)
        env = self.environment.build_env()
        return MultiStepWrapper(
            env, self.n_obs_steps, self.n_action_steps,
            max_episode_steps=self.max_episode_steps, obs_transform=obs_transform,
        )

    def rollout(self, n_episodes: int) -> list[float]:
        if self.n_eval_envs > 1:
            return self.rollout_vectorized(n_episodes)
        return self.rollout_serial(n_episodes)

    def rollout_serial(self, n_episodes: int) -> list[float]:
        policy = self.make_policy()
        wrapper = self.make_rollout_wrapper()
        scores: list[float] = []
        for ep in range(n_episodes):
            obs = wrapper.reset(seed=self.rollout_seed + ep)
            done = False
            while not done:
                chunk = policy.predict_action(obs)
                obs, _reward, done, _info = wrapper.step(chunk.detach().cpu().numpy())
            scores.append(wrapper.episode_score())
        return scores

    def rollout_vectorized(self, n_episodes: int) -> list[float]:
        policy = self.make_policy()
        n_envs = min(self.n_eval_envs, n_episodes)
        if self.eval_wrappers is None:
            self.eval_wrappers = [self.make_rollout_wrapper() for _ in range(n_envs)]
        scores: list[float] = []
        ep = 0
        while ep < n_episodes:
            wave = range(ep, min(ep + n_envs, n_episodes))
            active = self.eval_wrappers[: len(wave)]
            obs = [w.reset(seed=self.rollout_seed + i) for w, i in zip(active, wave)]
            while not all(w.done for w in active):
                chunks = policy.predict_action(np.stack(obs))            # (N, To, Do)
                obs = [w.step(np.asarray(chunks[k].detach().cpu()))[0]
                       for k, w in enumerate(active)]
            scores.extend(w.episode_score() for w in active)
            ep += len(wave)
        return scores

    # ── evaluation (replaces the distribution-sampling eval) ───────────────────
    def evaluate(self, step: int = 0) -> list[float]:
        if self.environment is None:
            raise ValueError("PolicyTrainingRunner.evaluate requires an environment adapter")
        if self.n_rollout_episodes <= 0:
            return []                       # eval intentionally disabled
        self.model.eval()
        with torch.no_grad():
            scores = self.rollout(self.n_rollout_episodes)
        metrics = _mean_score(scores, self.success_threshold)
        print(f"[eval] step {step} · episodes {len(scores)} · "
              f"test/mean_score={metrics['mean']:.4f}")
        self.model.train()
        return scores


__all__ = ["PolicyTrainingRunner"]

"""Guided receding-horizon locomotion rollout on forge's planning stack — the forge port of
the reference implementation's ``GuidedPlanningPolicy`` + ``PlanningRunner.rollout`` (diffuser locomotion protocol).

Protocol (per env step): pin the current normalized state at trajectory position 0 (obs columns
only), sample ``n_samples`` candidate ``[a | s]`` plans under EMA weights with value-gradient
guidance during denoising, rank candidates by ``V(plan, t=0)``, execute the first action of the
best plan, replan. Episode score = sum of env rewards (D4RL return convention).

NORMALIZATION-MISMATCH NOTE (deliberate, for checkpoint parity — do NOT "fix"):
The reference implementation trained the locomotion PLANNERS under a gaussian (standardize) trajectory normalizer but
the VALUE heads under a minmax one, and its eval code (``ValueGuidanceConstraint.apply`` and the
plan ranking in ``GuidedPlanningPolicy``) feeds planner-space (gaussian-normalized) trajectories
straight into the minmax-trained value net with NO conversion. The trained checkpoints' behavior
— and the reference implementation's actual benchmark numbers — bake in that quirk, so this port reproduces it
verbatim: ``DiffuserValueGuidance`` and the ranking below evaluate V on gaussian-normalized
plans. If the value heads are ever retrained under the planner's normalizer, delete this note.
"""
from __future__ import annotations

import numpy as np
import torch

from forge.core.registry import register
from forge.control.value_guidance import AmortizedValueController
from forge.runners.multistep import MultiStepWrapper
from forge.utils.seeding import make_generator


@register("control", "diffuser_value_guidance")
class DiffuserValueGuidance(AmortizedValueController):
    """The reference implementation’s ``ValueGuidanceConstraint`` semantics as a forge x0-surface controller.

    Differences from forge's built-in ``value_guidance`` (kept for parity with the reference implementation):
      - V is conditioned on the CURRENT diffusion time t (value nets are noise-level aware);
      - ``n_guide_steps`` ascent nudges per reverse step (diffuser: 2), not 1;
      - guidance freezes once ``t < t_stopgrad / n_steps`` (diffuser's discrete cutoff);
      - step size = ``scale * sigma(t)^2`` (the reference implementation's continuous proxy for the posterior var).
    """

    surface = "x0"

    def __init__(self, value_checkpoint: str, *, scale: float = 0.1, n_guide_steps: int = 2,
                 t_stopgrad: int = 2, n_steps: int = 20, sigma_weight: bool = True):
        super().__init__(value_checkpoint, scale=scale, sigma_weight=sigma_weight, sign=1.0)
        self.n_guide_steps = int(n_guide_steps)
        self.t_stopgrad = int(t_stopgrad)
        self.n_steps = int(n_steps)

    def _load(self) -> None:
        # the reference implementation's CheckpointValueModel evaluates the RAW value weights (payload["model"]),
        # NOT the EMA shadow — forge's base controller prefers ema_state, which changes the
        # guidance gradients. Load model_state explicitly for parity.
        if self._value is not None:
            return
        from forge.core import registry
        from forge.core.checkpoint import load_checkpoint

        ckpt = load_checkpoint(self.value_checkpoint)
        mcfg = (ckpt.get("config") or {})["model"]
        model = registry.create("model", mcfg["name"], **(mcfg.get("params") or {}))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._value = model

    def _ensure(self, device) -> None:
        self._load()
        p = next(self._value.parameters())
        if p.device != torch.device(device):
            self._value.to(device)

    def value(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """V(x, t) -> (B,). x is in the PLANNER's normalized space (see module note)."""
        self._ensure(x.device)
        return self._value(x, t).reshape(-1)

    def modify_x0(self, x0_hat: torch.Tensor, x: torch.Tensor, t, schedule,
                  cond=None, context=None) -> torch.Tensor:
        # cond/context added to the base modify_x0 contract in genforge 0.1.1; this
        # value guidance is unconditional (ignores them), matching forge's own base impl.
        self._ensure(x0_hat.device)
        t_flat = torch.as_tensor(t, device=x0_hat.device, dtype=torch.float32).reshape(-1)
        if float(t_flat[0]) < self.t_stopgrad / self.n_steps:
            return x0_hat                              # frozen near the end of the chain

        step = self.scale
        if self.sigma_weight:
            step = step * float(schedule.sigma(t_flat[:1]).square())

        tt = t_flat.expand(x0_hat.shape[0]) if t_flat.numel() == 1 else t_flat
        z = x0_hat
        for _ in range(self.n_guide_steps):
            with torch.enable_grad():                  # samplers run under no_grad
                inp = z.detach().clone().requires_grad_(True)
                v = self._value(inp, tt)
                (g,) = torch.autograd.grad(v.sum(), inp)
            z = z + step * g.detach()
        return z


class GuidedLocomotionEval:
    """Best-of-N replanning rollout over a start-pinned trajectory sampler (forge parts).

    Args:
        runner:      a forge runner rebuilt from a transplanted planner checkpoint
                     (provides sampler, preprocessor, dataset, environment adapter, ema).
        value_ckpt:  transplanted value-head checkpoint; None = unguided best-of-N is
                     ranked trivially (first plan), matching the reference implementation with no constraint.
        action_dim:  width of the action slice in x = [a | s].
        n_samples:   candidate plans per env step (diffuser batch of 64).
        max_episode_steps / rollout_seed: the reference implementation locomotion eval defaults.
    """

    def __init__(self, runner, value_ckpt: str | None, *, action_dim: int,
                 n_samples: int = 64, max_episode_steps: int = 1000,
                 rollout_seed: int = 100000, sample_seed: int | None = None):
        self.runner = runner
        self.action_dim = int(action_dim)
        self.n_samples = int(n_samples)
        self.max_episode_steps = int(max_episode_steps)
        self.rollout_seed = int(rollout_seed)
        # Seeding the ENV alone is not reproducibility: the plans are drawn by the reverse process,
        # so an unseeded sampler re-rolls every candidate each run and the SAME checkpoint scores
        # differently. Defaults to the runner's own sample_seed — the seed forge already uses for
        # `TrainingRunner.sample` — so one knob controls both.
        seed = getattr(runner, "sample_seed", None) if sample_seed is None else sample_seed
        self.sample_seed = None if seed is None else int(seed)   # None = unseeded (matches PolicyWrapper)
        self._gen: "torch.Generator | None" = None     # set per EPISODE by rollout()
        self.control = None
        if value_ckpt is not None:
            self.control = DiffuserValueGuidance(
                str(value_ckpt), n_steps=int(runner.n_sample_steps))
            runner.sampler.control = self.control

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize the obs slice with the PLANNER's (gaussian) stats — cols action_dim:."""
        pre = self.runner.preprocessor
        if pre is None:
            return obs
        mean = torch.as_tensor(pre.mean, device=obs.device)[self.action_dim:]
        std = torch.as_tensor(pre.std, device=obs.device)[self.action_dim:]
        return (obs - mean) / std.clamp_min(getattr(pre, "eps", 1e-6))

    @torch.no_grad()
    def predict_action(self, obs_stack) -> np.ndarray:
        r = self.runner
        device = torch.device(r.device)
        h, dim = r.dataset.sample_shape

        obs = torch.as_tensor(np.asarray(obs_stack), dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs = obs[-1:].to(device)                      # current state (1, Do)
        nobs = self._normalize_obs(obs)

        # pin the obs columns of trajectory row 0 (the reference implementation cond {0: obs})
        mask = torch.zeros(h, dim, dtype=torch.bool)
        mask[0, self.action_dim:] = True
        values = torch.zeros(self.n_samples, h, dim, device=device)
        values[:, 0, self.action_dim:] = nobs

        out = r.sampler.sample((self.n_samples, h, dim), r.n_sample_steps,
                               cond={"inpaint": (mask, values)}, generator=self._gen)
        plans = out.samples                            # (N, H, dim), planner-normalized

        best = 0
        if self.control is not None and plans.shape[0] > 1:
            t0 = torch.zeros(plans.shape[0], device=plans.device)
            best = int(self.control.value(plans, t0).argmax())   # diffuser sort_by_values

        plan = plans[best:best + 1]
        if r.preprocessor is not None:
            plan = r.preprocessor.inverse(plan)
        return plan[0, 0, :self.action_dim].detach().cpu().numpy()

    def rollout(self, n_episodes: int) -> list[float]:
        r = self.runner
        device = torch.device(r.device)
        r.model.to(device)
        r.model.eval()
        used_ema = r.ema is not None                   # eval under EMA weights, like the reference implementation
        if used_ema:
            r.ema.store(r.model)
            r.ema.copy_to(r.model)
        if self.control is not None:
            self.control.prepare(r.preprocessor)

        wrapper = MultiStepWrapper(r.environment.build_env(), 1, 1,
                                   max_episode_steps=self.max_episode_steps,
                                   reward_agg="sum")
        try:
            scores: list[float] = []
            for ep in range(n_episodes):
                obs = wrapper.reset(seed=self.rollout_seed + ep)
                # One generator per EPISODE, derived from (sample_seed, ep): plan noise still varies
                # across replans and across episodes, but the whole rollout is reproducible. Made
                # here, not in __init__, so episode e does not inherit the generator POSITION left
                # by however many replans episode e-1 happened to need.
                self._gen = make_generator(self.sample_seed + ep, device)
                done = False
                while not done:
                    action = self.predict_action(obs)
                    obs, _, done, _ = wrapper.step(action[None, :])
                scores.append(wrapper.episode_score())
            return scores
        finally:
            wrapper.close()
            if used_ema:
                r.ema.restore(r.model)

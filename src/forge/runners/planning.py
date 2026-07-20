"""Planning runner: train a trajectory diffuser, then produce goal-conditioned plans.

Reuses TrainingRunner's training loop verbatim; planning is goal-conditioned sampling — pin the
start/goal timesteps (in normalized space) every reverse step via the inpainting cond, then invert
the membrane. `evaluate` plans between in-distribution endpoints and reports endpoint error
(endpoints are pinned exactly in normalized space; the raw-unit endpoint matches to floating-point
precision ~1e-6 after the affine inverse), path contiguity, and constraint feasibility.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.registry import register
from .training import TrainingRunner


@register("runner", "planning")
class PlanningRunner(TrainingRunner):
    def _normalize(self, p: torch.Tensor) -> torch.Tensor:
        return self.preprocessor.transform(p) if self.preprocessor is not None else p

    def plan(self, starts: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        """Plan trajectories connecting each (start, goal) pair. Inputs/outputs in raw units.

        The pinned timesteps are taken from the trained method's ``pin_positions`` (start = first,
        goal = last) so train-time and plan-time pinning cannot drift apart."""
        starts = torch.as_tensor(starts, dtype=torch.float32).reshape(-1, self.dataset.dim)
        goals = torch.as_tensor(goals, dtype=torch.float32).reshape(-1, self.dataset.dim)
        h, dim = self.dataset.sample_shape
        ns, ng = self._normalize(starts), self._normalize(goals)

        pins = self.method.pin_positions        # no default: a planner needs a pinning method
        start_idx, goal_idx = pins[0] % h, pins[-1] % h
        if start_idx == goal_idx:
            raise ValueError(
                f"plan() pins a start AND a goal, but method.pin_positions={tuple(pins)} names a "
                f"single timestep ({start_idx} of horizon {h}) — the goal would silently overwrite "
                f"the start and every plan would begin AT the goal. A start-pin-only model cannot "
                f"honor a goal: pair it with a rollout runner, or train with two pins."
            )
        mask = torch.zeros(h, dim, dtype=torch.bool)
        mask[start_idx] = True
        mask[goal_idx] = True
        values = torch.zeros(starts.shape[0], h, dim)
        values[:, start_idx, :] = ns
        values[:, goal_idx, :] = ng
        return self.sample(n=starts.shape[0], cond={"inpaint": (mask, values)})

    def evaluate(self, n_pairs: int = 16) -> dict:
        p = min(n_pairs, self.dataset.num_items)
        windows = self.dataset.gather(torch.arange(p))        # (P, H, dim) raw, in-distribution
        starts, goals = windows[:, 0], windows[:, -1]

        plans = self.plan(starts, goals)                      # (P, H, dim) raw, on sampler device
        starts, goals = starts.to(plans.device), goals.to(plans.device)  # dataset is CPU; sampler may be CUDA
        endpoint_err = (
            (plans[:, 0] - starts).norm(dim=-1) + (plans[:, -1] - goals).norm(dim=-1)
        ).mean()
        max_step = (plans[:, 1:] - plans[:, :-1]).norm(dim=-1).max()
        metrics = {
            "endpoint_error": float(endpoint_err.item()),
            "max_step": float(max_step.item()),
            "n_pairs": float(p),
        }

        control = getattr(self.sampler, "control", None)
        raw_cost = getattr(control, "_raw_cost", None) if control is not None else None
        if raw_cost is not None and hasattr(raw_cost, "feasible"):
            metrics["feasible"] = float(raw_cost.feasible(plans.reshape(-1, self.dataset.dim)).float().mean().item())

        # This override replaces TrainingRunner.evaluate wholesale, so a configured `metric` and the
        # samples/metrics artifacts are only honored if we do it here too — otherwise `+metric=...`
        # is built, injected, and silently produces nothing (the design contract: the runner persists both).
        # Planning has no held-out split, so pass held_out=None (a data-driven metric would need one).
        if self.metric is not None:
            metrics.update(self.metric(samples=plans, held_out=None))
        self._persist_eval(plans, metrics)          # shared persistence — stays in lockstep with base eval
        return metrics

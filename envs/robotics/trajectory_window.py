"""Family-agnostic fixed-length sliding-window dataset over raw trajectory episodes.

Built on the genforge ``BaseDataset``
surface (``gather`` / ``batch`` / ``fit_tensor``). Episodes are concatenated once into a flat
``(sumT, x_dim)`` tensor; windows are gathered on the fly from per-window start indices, clamped
to each window's episode bounds. The materialized ``(num_windows, H, x_dim)`` tensor is never
built (the flat-tensor memory win, mirrored from ``envs/trajectory_synth/dataset.py``).

Layouts (the generated quantity ``x`` the membrane normalizes — Inv 9):
  - "janner"        : x = [actions | observations]   (include_actions implied True)
  - "actions_only"  : x = actions; observations ride alongside as model conditioning (cond)
  - "obs_only"      : x = observations[:, target_obs_dims]   (actions ignored)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from forge.core.protocols import BaseDataset, BatchProtocol
from forge.core.registry import register


@register("dataset", "trajectory_window")
class TrajectoryWindowDataset(BaseDataset):
    def __init__(
        self,
        environment,
        horizon: int,
        *,
        layout: str = "janner",
        stride: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        include_actions: bool = True,
        target_obs_dims: Sequence[int] | None = None,
        value_targets: bool = False,
        discount: float = 0.99,
        max_windows: int | None = None,
        n_obs_steps: int | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        if layout not in ("janner", "actions_only", "obs_only"):
            raise ValueError(f"layout must be janner|actions_only|obs_only, got {layout!r}")
        if pad_before < 0 or pad_after < 0:
            raise ValueError(f"pad_before/pad_after must be >= 0, got {pad_before}/{pad_after}")
        if value_targets and (pad_before > 0 or pad_after > 0):
            # window→episode-suffix returns are undefined under edge-replication.
            raise ValueError("value_targets=True is incompatible with pad_before/pad_after")

        self.horizon = int(horizon)
        self.stride = int(stride)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.layout = layout
        self.include_actions = layout == "janner" or bool(include_actions)
        self.value_targets = bool(value_targets)
        self.discount = float(discount)
        self.target_obs_dims = (
            None if target_obs_dims is None else tuple(int(d) for d in target_obs_dims)
        )
        if self.target_obs_dims is not None and not self.target_obs_dims:
            raise ValueError("target_obs_dims cannot be empty")

        # n_obs_steps: emit cond as the first-To obs frames flattened to (B, To*obs_dim) — the
        # diffusion-policy global conditioning vector PolicyWrapper feeds at rollout (Inv 9).
        self.n_obs_steps = None if n_obs_steps is None else int(n_obs_steps)
        if self.n_obs_steps is not None:
            if self.layout != "actions_only":
                raise ValueError(
                    f"n_obs_steps obs-history conditioning only applies to layout='actions_only', "
                    f"got {self.layout!r}"
                )
            if self.n_obs_steps < 1 or self.n_obs_steps > self.horizon:
                raise ValueError(
                    f"n_obs_steps must be in [1, horizon={self.horizon}], got {self.n_obs_steps}"
                )

        # ── extract episodes (trimmed: no image/dict-obs handling) ────────────────────────────
        def _arr(ep, key):
            if key not in ep:
                return None
            a = np.asarray(ep[key])
            return None if a.ndim == 0 else torch.as_tensor(a, dtype=torch.float32)

        def _obs(ep):
            if "observations" not in ep:
                raise ValueError(f"episode missing 'observations'; keys={list(ep.keys())}")
            o = torch.as_tensor(np.asarray(ep["observations"]), dtype=torch.float32)
            if o.ndim != 2:
                raise ValueError(f"expected obs rank-2 (T, D), got {tuple(o.shape)}")
            return o

        def _suffix_returns(ep, T):
            # Discounted suffix returns S[t] = Σ_{k≥t} γ^{k-t} r_k. float64:
            # γ^t spans ~1e-5..1 over long episodes, so the divide-back stays well-conditioned.
            rewards = _arr(ep, "rewards")
            if rewards is None:
                raise ValueError("value_targets=True requires episodes with a 'rewards' key")
            rewards = rewards.reshape(-1)[:T].to(torch.float64)
            w = torch.pow(
                torch.tensor(self.discount, dtype=torch.float64),
                torch.arange(T, dtype=torch.float64),
            )
            tail = torch.flip(torch.cumsum(torch.flip(w * rewards, [0]), 0), [0])
            return (tail / w).to(torch.float32)

        # ── windowing → flat tensors + per-window index/bound tensors ────────────────────────
        flats: list[torch.Tensor] = []
        obs_flats: list[torch.Tensor] = []
        starts: list[int] = []
        lo: list[int] = []
        hi: list[int] = []
        values: list[float] = []
        run = 0
        obs = None  # keep in scope for dim bookkeeping after loop
        actions = None
        for ep in environment.episodes():
            if max_windows is not None and len(starts) >= max_windows:
                break
            obs = _obs(ep)
            actions = _arr(ep, "actions")
            T = obs.shape[0] if actions is None else min(obs.shape[0], actions.shape[0])
            obs, actions = obs[:T], (None if actions is None else actions[:T])

            xep = self._episode_x(obs, actions)             # (T, x_dim) in the chosen layout
            suffix = _suffix_returns(ep, T) if self.value_targets else None

            ep_global = run
            flats.append(xep)
            if self.layout == "actions_only":
                obs_flats.append(obs)                       # obs ride alongside as conditioning
            run += T

            # anchor semantics: starts in [-pad_before, T - horizon + pad_after].
            for start in range(-self.pad_before, T - self.horizon + self.pad_after + 1, self.stride):
                starts.append(ep_global + start)
                lo.append(ep_global)
                hi.append(ep_global + T - 1)
                if suffix is not None:
                    values.append(float(suffix[min(max(start, 0), T - 1)]))
                if max_windows is not None and len(starts) >= max_windows:
                    break

        if not starts:
            raise ValueError(
                f"no windows produced (horizon={horizon}, stride={stride}); "
                "every episode is shorter than horizon - pad_before - pad_after."
            )

        self.flat = torch.cat(flats, dim=0)                                   # (sumT, x_dim)
        self.flat_obs = torch.cat(obs_flats, dim=0) if obs_flats else None    # (sumT, obs_dim)|None
        self.window_starts = torch.tensor(starts, dtype=torch.long)
        self.window_lo = torch.tensor(lo, dtype=torch.long)
        self.window_hi = torch.tensor(hi, dtype=torch.long)
        self.window_offsets = torch.arange(self.horizon, dtype=torch.long)
        self.window_values = (
            torch.tensor(values, dtype=torch.float32) if self.value_targets else None
        )

        # dimension bookkeeping.
        self.obs_dim = obs.shape[-1]
        self.action_dim = 0 if actions is None else actions.shape[-1]
        self.x_dim = self.flat.shape[-1]

    # ── layout assembly ──────────────────────────────────────────────────────────────────────
    def _episode_x(self, obs: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        if self.layout == "actions_only":
            if actions is None:
                raise ValueError("layout='actions_only' but episode has no 'actions'")
            return actions.contiguous()
        obs_t = obs if self.target_obs_dims is None else obs[:, list(self.target_obs_dims)]
        if self.layout == "obs_only":
            return obs_t.contiguous()
        # janner: x = [actions | observations]
        if actions is None:
            raise ValueError("layout='janner' but episode has no 'actions'")
        return torch.cat([actions, obs_t], dim=-1).contiguous()

    # ── flat gather with episode-bound clamp (mirrors
    #    envs/trajectory_synth/dataset.py L51-59) ──────────────────────────────────────────────
    def _gather_idx(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(self.window_starts.device)
        g = self.window_starts[idx][:, None] + self.window_offsets[None, :]      # (B, H)
        return g.clamp(self.window_lo[idx][:, None], self.window_hi[idx][:, None])

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Materialize a batch of x-windows ``(B, H, x_dim)`` on the fly from the flat tensor."""
        g = self._gather_idx(idx)
        return self.flat.to(g.device)[g]

    def _obs_history_cond(self, g: torch.Tensor) -> torch.Tensor:
        """Flatten the first n_obs_steps obs frames of each gathered window to (B, To*obs_dim) —
        the diffusion-policy global conditioning vector (matches PolicyWrapper at rollout, Inv 9).
        Edge-replication of g already mirrors MultiStepWrapper's obs warm-up at episode starts."""
        to = self.n_obs_steps
        obs_win = self.flat_obs.to(g.device)[g[:, :to]]                          # (B, To, obs_dim)
        return obs_win.reshape(g.shape[0], to * self.obs_dim)

    def batch(self, idx: torch.Tensor) -> BatchProtocol:
        """BatchProtocol entry point. actions_only attaches obs windows as cond (Inv 9):
        a flat (B, To*obs_dim) obs-history vector when n_obs_steps is set (policy conditioning),
        else the full (B, H, obs_dim) window. value_targets attaches per-window discounted returns."""
        x0 = self.gather(idx)
        cond = None
        if self.layout == "actions_only" and self.flat_obs is not None:
            g = self._gather_idx(idx)
            if self.n_obs_steps is not None:
                cond = self._obs_history_cond(g)                                 # (B, To*obs_dim)
            else:
                cond = self.flat_obs.to(g.device)[g]                             # (B, H, obs_dim)
        elif self.window_values is not None:
            cond = self.window_values[idx.to(self.window_values.device)].unsqueeze(-1)
        return BatchProtocol(x0=x0, cond=cond)

    @property
    def cond_fit_tensor(self) -> torch.Tensor | None:
        """Flattened first-n_obs_steps obs over all windows (num_items, To*obs_dim) — the fit data
        for the model's ObsNormalizer (Inv 9). None unless actions_only obs-history conditioning
        is active (the legacy (B,H,obs_dim) cond path has no in-model normalizer to fit)."""
        if self.n_obs_steps is None or self.layout != "actions_only" or self.flat_obs is None:
            return None
        return self._obs_history_cond(self._gather_idx(torch.arange(self.num_items)))

    @property
    def dim(self) -> int:
        """Alias for x_dim — total generated-quantity width (actions + obs for janner layout)."""
        return self.x_dim

    # ── BaseDataset surface ──────────────────────────────────────────────────────────────────
    @property
    def fit_tensor(self) -> torch.Tensor:
        return self.flat

    @property
    def num_items(self) -> int:
        return self.window_starts.shape[0]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return (self.horizon, self.x_dim)

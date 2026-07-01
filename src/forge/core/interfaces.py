"""Component contracts — every component ABC.

These are the signatures the rest of the framework programs against; concrete `@register`ed
implementations live in their respective modules.

Invariants these contracts encode:
  - The continuous/discrete distinction lives ONLY in `Space` and `Schedule` (Invariant 1).
  - Output-type conversion (eps ↔ x0 ↔ score) lives ONLY in `Schedule` (Invariant 3).
  - Everything inside the preprocessor membrane is in normalized coordinates (Invariant 2).
  - Dependencies are injected at construction, not threaded per call (Invariant 4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from ..utils.torch_utils import expand_like as _expand

if TYPE_CHECKING:  # keep heavy/optional imports out of the hot import path
    from torch import Generator, Tensor


# ── space.py ──────────────────────────────────────────────────────────────────────────────────
class Space(ABC):
    """State space + forward kernel. The ONLY home of the continuous-vs-discrete distinction."""

    @abstractmethod
    def prior_sample(self, shape, generator: "Optional[Generator]" = None) -> "Tensor":
        """Sample x_0 ~ p_0 (the reference prior)."""
        raise NotImplementedError

    @abstractmethod
    def forward_sample(
        self, x0: "Tensor", t: "Tensor", schedule: "Schedule",
        generator: "Optional[Generator]" = None,
    ) -> "Tensor":
        """Sample x_t ~ q(x_t | x_0). The forward primitive (training)."""
        raise NotImplementedError


# ── schedule.py ───────────────────────────────────────────────────────────────────────────────
class Schedule(ABC):
    """Time/noise schedule. With `Space`, the only place cont/disc may diverge (Invariant 1)."""

    @abstractmethod
    def discretize(self, n_steps: int) -> "Tensor":
        """Time grid for sampling (1→0 for diffusion, 0→1 for flow)."""
        raise NotImplementedError


class ContinuousSchedule(Schedule):
    """Continuous schedule + the shared output-type conversion math (Invariant 3 lives here)."""

    @abstractmethod
    def alpha(self, t: "Tensor") -> "Tensor":
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: "Tensor") -> "Tensor":
        raise NotImplementedError

    @abstractmethod
    def G(self, t: "Tensor") -> "Tensor":
        """Diffusion coefficient G_t."""
        raise NotImplementedError

    @abstractmethod
    def marginal(self, x0: "Tensor", t: "Tensor") -> "tuple[Tensor, Tensor]":
        """(mean, std) of q(x_t | x_0)."""
        raise NotImplementedError

    # Output-type conversion — THE shared α/σ math. Lives here, nowhere else (Invariant 3).
    # Affine marginals q(x_t|x_0)=N(α x_0, σ²) give ONE closed form for every continuous schedule,
    # so the conversions are concrete on the base and samplers/methods never branch on output_type.
    # A non-affine ContinuousSchedule would override these.
    def _coeffs(self, t: "Tensor", ref: "Tensor") -> "tuple[Tensor, Tensor]":
        """(α, σ) broadcast to ``ref``'s trailing dims — the shared affine coefficients."""
        return _expand(self.alpha(t), ref), _expand(self.sigma(t), ref)

    def score_from_eps(self, xt: "Tensor", eps: "Tensor", t: "Tensor") -> "Tensor":
        # score = ∇ log N(x_t; α x_0, σ²) = −(x_t − α x_0)/σ² = −ε/σ
        _, s = self._coeffs(t, xt)
        return -eps / s

    def score_from_x0(self, xt: "Tensor", x0: "Tensor", t: "Tensor") -> "Tensor":
        a, s = self._coeffs(t, xt)
        return -(xt - a * x0) / s**2

    def x0_from_eps(self, xt: "Tensor", eps: "Tensor", t: "Tensor") -> "Tensor":
        a, s = self._coeffs(t, xt)
        return (xt - s * eps) / a

    def x0_from_score(self, xt: "Tensor", score: "Tensor", t: "Tensor") -> "Tensor":
        a, s = self._coeffs(t, xt)
        return (xt + s**2 * score) / a

    def eps_from_x0(self, xt: "Tensor", x0: "Tensor", t: "Tensor") -> "Tensor":
        a, s = self._coeffs(t, xt)
        return (xt - a * x0) / s

    # Velocity conversions — used by flow paradigms; a VP/score schedule may leave these unset.
    def x0_from_velocity(self, xt: "Tensor", v: "Tensor", t: "Tensor") -> "Tensor":
        raise NotImplementedError

    def velocity_from_x0(self, xt: "Tensor", x0: "Tensor", t: "Tensor") -> "Tensor":
        raise NotImplementedError

    # ── Control-surface relations (Invariant 6 drift surface). ──────────────────────────────────
    # The control "drift" is the deterministic DATA-WARD heading of the reverse process — the rate
    # at which the iterate moves toward its clean estimate, ẋ ∝ (x̂₀ − xₜ). Chosen over the raw
    # probability-flow velocity because it is (a) exactly invertible and (b) data-ward with a
    # uniform +1 slope dx̂₀/ddrift for EVERY paradigm (the PF velocity's sign flips between score and
    # flow), so a drift filter (CBF/MPPI) acts in the correct direction without per-schedule signs.
    # The per-step α/σ scaling of the true b_θ is reapplied downstream by the sampler's posterior.
    def drift_from_x0(self, xt: "Tensor", x0: "Tensor", t: "Tensor") -> "Tensor":
        return x0 - xt

    def x0_from_drift(self, xt: "Tensor", drift: "Tensor", t: "Tensor") -> "Tensor":
        return xt + drift

    def reverse_drift(self, xt: "Tensor", score: "Tensor", t: "Tensor") -> "Tensor":
        """Assemble the reverse drift (heading) from the score, via the implied clean estimate."""
        return self.x0_from_score(xt, score, t) - xt

    # ── Output-type DISPATCH (still inside the schedule — Invariant 3). ──────────────────────────
    # Samplers/methods call these with `model.output_type`; the schedule selects the right
    # conversion. The caller never does conversion math and never branches on output_type itself.
    def x0_from(self, output_type: str, xt: "Tensor", pred: "Tensor", t: "Tensor") -> "Tensor":
        if output_type == "x0":
            return pred
        if output_type == "eps":
            return self.x0_from_eps(xt, pred, t)
        if output_type == "score":
            return self.x0_from_score(xt, pred, t)
        if output_type == "velocity":
            return self.x0_from_velocity(xt, pred, t)
        raise ValueError(f"Unsupported output_type {output_type!r} for x0_from.")

    def score_from(self, output_type: str, xt: "Tensor", pred: "Tensor", t: "Tensor") -> "Tensor":
        if output_type == "score":
            return pred
        if output_type == "eps":
            return self.score_from_eps(xt, pred, t)
        if output_type == "x0":
            return self.score_from_x0(xt, pred, t)
        if output_type == "velocity":
            return self.score_from_x0(xt, self.x0_from_velocity(xt, pred, t), t)
        raise ValueError(f"Unsupported output_type {output_type!r} for score_from.")

    def eps_from(self, output_type: str, xt: "Tensor", pred: "Tensor", t: "Tensor") -> "Tensor":
        if output_type == "eps":
            return pred
        return self.eps_from_x0(xt, self.x0_from(output_type, xt, pred, t), t)

    def velocity_from(self, output_type: str, xt: "Tensor", pred: "Tensor", t: "Tensor") -> "Tensor":
        if output_type == "velocity":
            return pred
        return self.velocity_from_x0(xt, self.x0_from(output_type, xt, pred, t), t)

    # min-SNR-γ cap on the x0-parametrization weight (Hang et al., 2023). Plain SNR weighting on
    # x0-prediction starves the high-noise gradients and under-learns coarse structure; capping the
    # SNR at γ rebalances it. ε stays unit-weighted (the reference parametrization), so this leaves
    # the ε-prediction objective — and its acceptance gate — unchanged.
    max_snr_weight: float = 5.0

    def loss_weight(self, output_type: str, t: "Tensor") -> "Tensor":
        """Per-sample weight that makes the weighted MSE on ``output_type`` comparable to the
        ε-prediction loss (SNR matching). This is α/σ math, so it lives in the schedule
        (Invariant 3): a method weights its loss by this and stays output-type-agnostic.
        eps/velocity → 1; x0 → min((α/σ)², γ); score → σ²."""
        t = torch.as_tensor(t, dtype=torch.float32)
        if output_type in ("eps", "velocity"):
            return torch.ones_like(t)
        a, s = self.alpha(t), self.sigma(t)
        if output_type == "x0":
            return torch.clamp((a / s) ** 2, max=self.max_snr_weight)
        if output_type == "score":
            return s**2
        raise ValueError(f"Unsupported output_type {output_type!r} for loss_weight.")

    def regression_target(
        self, output_type: str, *, x0: "Tensor", eps: "Tensor", xt: "Tensor", t: "Tensor"
    ) -> "Tensor":
        """The training target a model with ``output_type`` should regress to, given the clean
        sample ``x0`` and the noise ``eps`` realized in ``xt``. (Output-type math stays here.)"""
        if output_type == "eps":
            return eps
        if output_type == "x0":
            return x0
        if output_type == "score":
            return self.score_from_eps(xt, eps, t)
        if output_type == "velocity":
            return self.velocity_from_x0(xt, x0, t)
        raise ValueError(f"Unsupported output_type {output_type!r} for regression_target.")


class DiscreteSchedule(Schedule):
    """Discrete (CTMC / D3PM) schedule. Discrete is always x0-prediction."""

    @abstractmethod
    def Q(self, t: "Tensor") -> "Tensor":
        """(V, V) one-step transition matrix."""
        raise NotImplementedError

    @abstractmethod
    def Qbar(self, t: "Tensor") -> "Tensor":
        """Cumulative Q̄_t."""
        raise NotImplementedError

    def reverse_probs(
        self, xt: "Tensor", t: "Tensor", s: "Tensor", x0_logits: "Tensor"
    ) -> "Tensor":
        """q(x_s | x_t) = Σ_{x0} q(x_s | x_t, x0) p_θ(x0 | x_t)."""
        raise NotImplementedError


# ── model.py ──────────────────────────────────────────────────────────────────────────────────
class Model(nn.Module, ABC):
    """A learned field. `output_type` tells the schedule how to interpret the output."""

    output_type: str  # "score" | "eps" | "x0" | "velocity" | "logits"

    @abstractmethod
    def forward(self, x: "Tensor", t: "Tensor", cond=None) -> "Tensor":
        raise NotImplementedError


# ── method.py ─────────────────────────────────────────────────────────────────────────────────
class Method(ABC):
    """A training objective. Deps injected at build (Invariant 4)."""

    def __init__(self, schedule: Schedule, space: Space):
        self.schedule = schedule
        self.space = space

    @abstractmethod
    def loss(self, model: Model, x0: "Tensor", cond=None, generator: "Optional[Generator]" = None) -> "Tensor":
        raise NotImplementedError


# ── sampler.py ────────────────────────────────────────────────────────────────────────────────
class Sampler(ABC):
    """Reverse-process integrator. Output-type-agnostic; calls schedule conversions (Invariant 3).
    Applies `control.modify(...)` when control is present (Invariant 6)."""

    def __init__(self, model: Model, schedule: Schedule, space: Space, control: "Optional[Controller]" = None):
        self.model = model
        self.schedule = schedule
        self.space = space
        self.control = control
        self._generator: "Optional[Generator]" = None  # set during sample(); read by stochastic steps

    @abstractmethod
    def step(self, x: "Tensor", t: "Tensor", s: "Tensor", cond=None) -> "Tensor":
        """One reverse increment x_t → x_s. Calls schedule conversions for output-type; calls
        control.modify(...) when control is not None."""
        raise NotImplementedError

    def _apply_control(self, x0_hat: "Tensor", x: "Tensor", t: "Tensor") -> "Tensor":
        """Apply the controller and return the (possibly bent) clean estimate x̂₀, dispatching on the
        controller's SURFACE (Invariant 6). The base model is never modified.

          - ``"x0"``    — shift x̂₀ directly (Projection / Guidance / ValueGuidance).
          - ``"drift"`` — materialize a genuine reverse drift b_θ from x̂₀, let the controller filter
            the rate (CBF / MPPI / drift-FBSDE), then convert back to an equivalent x̂₀. The
            conversion is pure α/σ algebra in the schedule, so every drift-controller sees a real b_θ
            regardless of the sampler's internal parameterization.
        """
        c = self.control
        if c is None:
            return x0_hat
        if c.surface == "x0":
            return c.modify_x0(x0_hat, x, t, self.schedule)
        drift = self.schedule.drift_from_x0(x, x0_hat, t)
        drift = c.modify_drift(drift, x, t, self.schedule)
        return self.schedule.x0_from_drift(x, drift, t)

    def _apply_conditioning(self, x: "Tensor", cond) -> "Tensor":
        """The PIN SLOT — pinning conditioned entries to observed values, routed through the `Pin`
        controller (pinning IS a hard equality constraint = control). Applied AFTER the primary control,
        every reverse step, so pins WIN. The pin lands on the SAMPLE (byte-exact;
        a pure x̂₀ pin would be re-noised — see `Pin`). Conditioning forms:
          - ``{"pin": (indices, values)}``  — fix feature columns (2-D point conditioning);
          - ``{"inpaint": (mask, values)}`` — fix masked entries (trajectory start/goal pinning).
        """
        from ..control.projection import Pin

        pin = Pin.from_cond(cond)
        return pin.project(x) if pin is not None else x

    def sample(
        self, shape, n_steps: int, cond=None,
        generator: "Optional[Generator]" = None, return_chain: bool = False,
    ):
        """Framework owns the loop: prior_sample → walk the discretized grid via `step` → return.

        Shared by every sampler; subclasses implement only `step`. (Inverse-transform back to raw
        units is the runner's job, not the sampler's — Invariant 2.)
        """
        from ..utils.torch_utils import model_device
        from .types import SamplerOutput

        device = model_device(self.model)
        self._generator = generator
        x = self.space.prior_sample(shape, generator=generator, device=device)
        x = self._apply_conditioning(x, cond)
        # Time grid stays on CPU: its scalars are 0-dim and re-homed by each consumer (models re-home
        # t; `expand_like` re-homes schedule coeffs), so per-step `float(s)` terminal checks don't
        # force a device→host sync on the GPU sampling path.
        grid = self.schedule.discretize(n_steps)

        chain = [x.clone()] if return_chain else None
        with torch.no_grad():
            for i in range(n_steps):
                x = self.step(x, grid[i], grid[i + 1], cond)
                if return_chain:
                    chain.append(x.clone())
        self._generator = None
        return SamplerOutput(
            samples=x, chain=torch.stack(chain, dim=0) if return_chain else None
        )


# ── cost.py ───────────────────────────────────────────────────────────────────────────────────
class Cost(ABC):
    """WHAT you tilt toward. `log_h` is the tilt's log-density."""

    @abstractmethod
    def log_h(self, x: "Tensor", t: "Tensor") -> "Tensor":
        raise NotImplementedError

    def to_normalized(self, preprocessor: "Preprocessor") -> "Cost":
        """Map a real-unit spec into normalized space (Invariant 8). Default: already unitless."""
        return self


# ── control.py ────────────────────────────────────────────────────────────────────────────────
class Controller(ABC):
    """HOW you approximate the tilt Q ∝ exp(log_h)·P. The thesis research surface.

    A controller declares which SURFACE it acts on (Invariant 6) and implements exactly ONE of
    `modify_drift` / `modify_x0`:
      - ``"drift"`` — the GENERAL surface. The correction enters the reverse drift b_θ; required
        whenever the tilt reads the rate ẋ (CBF, MPPI, drift-FBSDE, Doob). No faithful x̂₀ form.
      - ``"x0"``    — a SPECIALIZATION: a shift of the clean estimate x̂₀ (Projection, Guidance,
        ValueGuidance). The sampler converts it up to an equivalent drift correction via the schedule.

    drift is primary because x̂₀→drift always converts, but drift→x̂₀ is lossy when the control reads
    the rate. The base model is frozen at sampling time; amortized controllers load their artifact in
    `prepare`, communicating with their paired `Method` ONLY through the checkpoint.
    """

    surface: str = "x0"

    def __init__(self, cost: "Optional[Cost]" = None):
        self._raw_cost = cost
        self.cost = cost

    def prepare(self, preprocessor: "Optional[Preprocessor]") -> None:
        """Map the cost spec into normalized coordinates once the membrane is known (Invariant 8).
        Idempotent — always derived from the original spec. Called by the runner after the
        preprocessor is fitted/loaded (the builder cannot, as stats don't exist at build time).
        Safe when the cost is unit-free (identity `to_normalized`) and when ``_raw_cost is None``
        (amortized controllers, which override this to also load their artifact)."""
        if preprocessor is not None and self._raw_cost is not None:
            self.cost = self._raw_cost.to_normalized(preprocessor)
        else:
            self.cost = self._raw_cost

    def modify_drift(self, drift: "Tensor", x: "Tensor", t: "Tensor", schedule: Schedule) -> "Tensor":
        """GENERAL surface (``surface == "drift"``). Filter / correct the reverse drift b_θ (e.g. a
        CBF QP, an MPPI reweighting). Reads the rate ẋ; not faithfully expressible on x̂₀."""
        raise NotImplementedError

    def modify_x0(self, x0_hat: "Tensor", x: "Tensor", t: "Tensor", schedule: Schedule) -> "Tensor":
        """SPECIALIZATION (``surface == "x0"``). Return a shifted clean estimate x̂₀; the sampler
        converts it to an equivalent drift correction via the schedule when needed."""
        raise NotImplementedError


# ── preprocessor.py ───────────────────────────────────────────────────────────────────────────
class Preprocessor(ABC):
    """The membrane. Fitted once, applied at train + sampling-init, inverted at output.

    Touches ONLY the generated quantity, never the conditioning (Invariant 9). Everything inside
    the membrane is normalized (Invariant 2). Stats travel in the checkpoint (Invariant 5).
    """

    @abstractmethod
    def fit(self, data: "Tensor") -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: "Tensor") -> "Tensor":
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x_tilde: "Tensor") -> "Tensor":
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, d: dict) -> None:
        raise NotImplementedError


# ── runner.py ─────────────────────────────────────────────────────────────────────────────────
class Runner(ABC):
    """Owns the lifecycle: fits the preprocessor, trains in normalized space, applies inverse on
    sample output, writes self-contained checkpoints (Invariant 5)."""

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> "Tensor":
        raise NotImplementedError

    def evaluate(self) -> dict:
        """Rollout / score where applicable. Optional; default no-op."""
        return {}

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

##-------------------------------------
class Space(ABC):
    """State space + forward kernel. The ONLY home of the continuous-vs-discrete distinction.
    Can be discrete (CTMC / D3PM) or continuous (diffusion / flow). Deps injected at build."""

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

##-------------------------------------
class Discretization(ABC):
    """HOW to space the t-grid for sampling (uniform, Karras/EDM, logSNR-uniform, ...).
    Orthogonal to the noise curve itself and agnostic to the cont/disc distinction,
    which remains owned solely by Space and Schedule (Invariant 1)."""

    @abstractmethod
    def grid(self, schedule: "Schedule", n_steps: int) -> "Tensor":
        raise NotImplementedError
 
class UniformDiscretization(Discretization):
    def grid(self, schedule, n_steps):
        # Endpoints/direction belong to the schedule (diffusion 1→0, flow 0→1). A schedule advertises
        # its sweep via `discretize`; absent one (a minimal stub), fall back to the diffusion default
        # — the pre-refactor behaviour. Delegate now; swap for endpoint-aware spacing
        # (Karras/EDM) when a schedule needs it.
        if hasattr(schedule, "discretize"):
            return schedule.discretize(n_steps)
        return torch.linspace(1.0, 0.0, n_steps + 1)
    
##--------------------------------------

class Schedule(ABC):
    """Time/marginal schedule. With `Space`, the only place cont/disc may diverge. 
    The convention followed is t=1 is known marginal and t=0 is clean data."""

    @abstractmethod
    def marginal(self, x0: "Tensor", t: "Tensor") -> "tuple[Tensor, Tensor]":
        """definition of q(x_t | x_0)."""
        raise NotImplementedError

class GaussianContinuousSchedule(Schedule):
    @abstractmethod
    def m(self, x0, t): raise NotImplementedError
    @abstractmethod
    def m_inverse(self, y, t): raise NotImplementedError
    @abstractmethod
    def L(self, t): raise NotImplementedError
    @abstractmethod
    def L_inverse(self, y, t): raise NotImplementedError
    @abstractmethod
    def m_dot(self, x0, t): raise NotImplementedError
    @abstractmethod
    def L_dot(self, t): raise NotImplementedError

    def marginal(self, x0, t): return self.m(x0, t), self.L(t)

    def eps_from_x0(self, xt, x0, t):
        return self.L_inverse(xt - self.m(x0, t), t)
    def score_from_x0(self, xt, x0, t):
        residual = xt - self.m(x0, t)
        return -self.L_inverse(self.L_inverse(residual, t), t)
    def score_from_eps(self, xt, eps, t):
        return -self.L_inverse(eps, t)
    def velocity_from_x0(self, xt, x0, t):
        eps = self.eps_from_x0(xt, x0, t)
        return self.m_dot(x0, t) + _expand(self.L_dot(t), eps) * eps

    def x0_from_eps(self, xt, eps, t):
        return self.m_inverse(xt - _expand(self.L(t), eps) * eps, t)
    def x0_from_score(self, xt, score, t):
        return self.m_inverse(xt + _expand(self.L(t), score) ** 2 * score, t)
    def x0_from_velocity(self, xt, v, t):
        raise NotImplementedError

    def drift_from_x0(self, xt, x0, t): return x0 - xt
    def x0_from_drift(self, xt, drift, t): return xt + drift
    def reverse_drift(self, xt, score, t): return self.x0_from_score(xt, score, t) - xt

    def x0_from(self, output_type, xt, pred, t):
        if output_type == "x0": return pred
        if output_type == "eps": return self.x0_from_eps(xt, pred, t)
        if output_type == "score": return self.x0_from_score(xt, pred, t)
        if output_type == "velocity": return self.x0_from_velocity(xt, pred, t)
        raise ValueError(output_type)
    def score_from(self, output_type, xt, pred, t):
        if output_type == "score": return pred
        if output_type == "eps": return self.score_from_eps(xt, pred, t)
        if output_type == "x0": return self.score_from_x0(xt, pred, t)
        if output_type == "velocity":
            return self.score_from_x0(xt, self.x0_from_velocity(xt, pred, t), t)
        raise ValueError(output_type)
    def eps_from(self, output_type, xt, pred, t):
        if output_type == "eps": return pred
        return self.eps_from_x0(xt, self.x0_from(output_type, xt, pred, t), t)
    def velocity_from(self, output_type, xt, pred, t):
        if output_type == "velocity": return pred
        return self.velocity_from_x0(xt, self.x0_from(output_type, xt, pred, t), t)
    def regression_target(self, output_type, *, x0, eps, xt, t):
        if output_type == "eps": return eps
        if output_type == "x0": return x0
        if output_type == "score": return self.score_from_eps(xt, eps, t)
        if output_type == "velocity": return self.velocity_from_x0(xt, x0, t)
        raise ValueError(output_type)

class AffineGaussianContinuousSchedule(GaussianContinuousSchedule):
    """m(x0,t) = α(t)x0, L(t) = σ(t). Only the affine-specific hooks live here."""

    @abstractmethod
    def alpha(self, t): raise NotImplementedError
    @abstractmethod
    def sigma(self, t): raise NotImplementedError
    @abstractmethod
    def alpha_dot(self, t): raise NotImplementedError
    @abstractmethod
    def sigma_dot(self, t): raise NotImplementedError

    def m(self, x0, t):        return _expand(self.alpha(t), x0) * x0
    def m_inverse(self, y, t): return y / _expand(self.alpha(t), y)
    def L(self, t):            return self.sigma(t)
    def L_inverse(self, y, t): return y / _expand(self.sigma(t), y)
    def m_dot(self, x0, t):    return _expand(self.alpha_dot(t), x0) * x0
    def L_dot(self, t):        return self.sigma_dot(t)

    def G(self, t):
        a, s, ad, sd = self.alpha(t), self.sigma(t), self.alpha_dot(t), self.sigma_dot(t)
        return torch.sqrt(2 * s**2 * (sd / s - ad / a))

    def x0_from_velocity(self, xt, v, t):
        a, s, ad, sd = self.alpha(t), self.sigma(t), self.alpha_dot(t), self.sigma_dot(t)
        a, s, ad, sd = (_expand(c, xt) for c in (a, s, ad, sd))
        return (v * s - sd * xt) / (ad * s - sd * a)

    max_snr_weight: float = 5.0

    def loss_weight(self, output_type, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        if output_type in ("eps", "velocity"): return torch.ones_like(t)
        a, s = self.alpha(t), self.sigma(t)
        if output_type == "x0": return torch.clamp((a / s) ** 2, max=self.max_snr_weight)
        if output_type == "score": return s**2
        raise ValueError(output_type)
    
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
##--------------------------------------

# ── model.py ──────────────────────────────────────────────────────────────────────────────────
class Model(nn.Module, ABC):
    """A learned field. `output_type` tells the schedule how to interpret the output."""

    output_type: str  # "score" | "eps" | "x0" | "velocity" | "logits"

    @abstractmethod
    def forward(self, x: "Tensor", t: "Tensor", cond=None) -> "Tensor":
        raise NotImplementedError

    def _check_cond(self, cond) -> None:
        """Shared guard for the ``(x, t, cond)`` contract (the design contract): a model with no conditioning
        pathway (``cond_dim`` == 0 or unset) must REJECT a model-conditioning tensor loudly instead
        of silently dropping it. Every model's ``forward`` calls this first, so the rule and message
        live in ONE place — a new model can't reintroduce the silent-drop bug.

        Only a *tensor* ``cond`` is model conditioning. The sampler threads its OWN conditioning spec
        (a dict, ``{"inpaint"/"pin": ...}``) through the same argument to the model; that is
        sampler-side (handled by ``_apply_conditioning``/``Pin``) and the model legitimately ignores
        it — so a dict ``cond`` never trips this guard. The complementary 'cond_dim>0 but cond=None'
        case is model-specific (an unconditional/CFG pass vs. a hard error) and stays in each model."""
        if torch.is_tensor(cond) and not getattr(self, "cond_dim", 0):
            raise ValueError(
                f"{type(self).__name__} has no conditioning pathway (cond_dim=0) but received a "
                "cond tensor; build it with cond_dim>0 or omit model conditioning."
            )


# ── criterion.py ──────────────────────────────────────────────────────────────────────────────
class Criterion(ABC):
    """A discrepancy measure between a model prediction and its regression target — the per-sample
    penalty (MSE, Huber, ...) a `Method` reduces to a scalar loss. Optional dependency injected at
    build (Invariant 4); a method with no configured criterion keeps its own built-in loss."""

    @abstractmethod
    def __call__(
        self, pred: "Tensor", target: "Tensor", weight: "Optional[Tensor]" = None
    ) -> "Tensor":
        raise NotImplementedError


# ── method.py ─────────────────────────────────────────────────────────────────────────────────
class Method(ABC):
    """A training objective. Deps injected at build (Invariant 4)."""

    def __init__(self, schedule: Schedule, space: Space, criterion: "Optional[Criterion]" = None):
        self.schedule = schedule
        self.space = space
        self.criterion = criterion

    @abstractmethod
    def loss(self, model: Model, x0: "Tensor", cond=None, generator: "Optional[Generator]" = None) -> "Tensor":
        raise NotImplementedError


# ── sampler.py ────────────────────────────────────────────────────────────────────────────────
class Sampler(ABC):
    """Reverse-process integrator. Output-type-agnostic; calls schedule conversions (Invariant 3).
    Applies `control.modify(...)` when control is present (Invariant 6)."""

    def __init__(self, model, schedule, space, control=None, discretization: "Discretization" = UniformDiscretization()):
        self.model = model
        self.schedule = schedule
        self.space = space
        self.control = control
        self.discretization = discretization
        self._check_absorbing_mask_agrees(schedule, space)
        self._generator: "Optional[Generator]" = None  # set during sample(); read by stochastic steps
        # Per-sample scratch a controller may read/write across steps (e.g. accumulate a running
        # statistic). Reset at the top of sample(), cleared at the end — single-use-per-call state,
        # same non-reentrant contract as _generator (do not share one Sampler across concurrent sample() calls).
        self._context: dict = {}

    @staticmethod
    def _check_absorbing_mask_agrees(schedule, space) -> None:
        """An absorbing schedule and its space MUST agree on which token is the mask.

        Build-time, and here because __init__ is the one place that holds both (the design contract:
        "Misconfigurations raise before a run wastes compute"). Duck-typed on the attribute, so this
        is a capability check, not a continuous-vs-discrete branch (Invariant 1).

        Why it must be loud: the schedule DERIVES a default (`num_classes - 1`) while the space
        leaves it None, so a config that sets one and not the other disagrees silently — the prior
        starts uniform instead of all-mask and the absorbing reverse, which only ever rewrites mask
        tokens, then freezes 100% of them in place. It samples, it reports, it is pure noise.
        """
        s_mask = getattr(schedule, "mask_index", None)
        if s_mask is None or not hasattr(space, "mask_index"):
            return                                    # not an absorbing pairing — nothing to agree on
        if space.mask_index != s_mask:
            raise ValueError(
                f"absorbing schedule mask_index={s_mask} but {type(space).__name__}.mask_index="
                f"{space.mask_index}. They must name the SAME token: the schedule only rewrites its "
                f"own mask, so a disagreeing space yields a prior it can never un-mask. Set "
                f"space.params.mask_index={s_mask} (the schedule defaults to num_classes-1)."
            )

    @abstractmethod
    def step(self, x: "Tensor", t: "Tensor", s: "Tensor", cond=None) -> "Tensor":
        """One reverse increment x_t → x_s. Calls schedule conversions for output-type; calls
        control.modify(...) when control is not None."""
        raise NotImplementedError

    def _apply_control(self, x0_hat: "Tensor", x: "Tensor", t: "Tensor", cond=None, context=None) -> "Tensor":
        """Apply the controller and return the (possibly bent) clean estimate x̂₀, dispatching on the
        controller's SURFACE (Invariant 6). The base model is never modified.

          - ``"x0"``    — shift x̂₀ directly (Projection / Guidance / ValueGuidance).
          - ``"drift"`` — materialize a genuine reverse drift b_θ from x̂₀, let the controller filter
            the rate (CBF / MPPI / drift-FBSDE), then convert back to an equivalent x̂₀. The
            conversion is pure α/σ algebra in the schedule, so every drift-controller sees a real b_θ
            regardless of the sampler's internal parameterization.

        ``cond`` (the sampler's conditioning spec) and ``context`` (per-sample scratch; defaults to the
        sampler's own ``self._context``) are threaded to the controller so it can steer per-conditioning
        and carry state across steps.
        """
        c = self.control
        if c is None:
            return x0_hat
        ctx = self._context if context is None else context
        if c.surface == "x0":
            return c.modify_x0(x0_hat, x, t, self.schedule, cond, ctx)
        drift = self.schedule.drift_from_x0(x, x0_hat, t)
        drift = c.modify_drift(drift, x, t, self.schedule, cond, ctx)
        return self.schedule.x0_from_drift(x, drift, t)

    def _apply_conditioning(self, x: "Tensor", cond) -> "Tensor":
        """The PIN SLOT — pinning conditioned entries to observed values, routed through the `Pin`
        controller (pinning IS a hard equality constraint = control). Applied AFTER the primary control,
        every reverse step, so pins WIN. The pin lands on the SAMPLE (byte-exact;
        a pure x̂₀ pin would be re-noised — see `Pin`). Conditioning forms:
          - ``{"pin": (indices, values)}``  — fix feature columns (2-D point conditioning);
          - ``{"inpaint": (mask, values)}`` — fix masked entries (trajectory start/goal pinning).
        """
        from .conditioning import Pin

        pin = Pin.from_cond(cond)
        return pin.project(x) if pin is not None else x

    def sample(self, shape, n_steps, cond=None, generator=None, return_chain=False):
        """Framework owns the loop: prior_sample → walk the discretized grid via `step` → return.

        Shared by every sampler; subclasses implement only `step`. (Inverse-transform back to raw
        units is the runner's job, not the sampler's — Invariant 2.)
        """
        from ..utils.torch_utils import model_device
        from ..core.types import SamplerOutput

        device = model_device(self.model)
        self._generator = generator
        self._context = {}                       # A3: fresh per-sample scratch for controllers
        x = self.space.prior_sample(shape, generator=generator, device=device)
        x = self._apply_conditioning(x, cond)    # pre-loop pin (also covers n_steps == 0)
        # Time grid stays on CPU: its scalars are 0-dim and re-homed by each consumer (models re-home
        # t; `expand_like` re-homes schedule coeffs), so per-step `float(s)` terminal checks don't
        # force a device→host sync on the GPU sampling path.
        grid = self.discretization.grid(self.schedule, n_steps) 

        chain = [x.clone()] if return_chain else None
        with torch.no_grad():
            for i in range(n_steps):
                x = self.step(x, grid[i], grid[i + 1], cond)
                # A4 defense-in-depth: pins must hold after EVERY reverse step. Continuous `step`s
                # already end with their own `_apply_conditioning` (pins win there), so this is
                # idempotent for them; it also guarantees the pin for any sampler whose `step` doesn't
                # self-pin. NOT a duplicate call — keep both (deleting either can silently drop a pin).
                x = self._apply_conditioning(x, cond)
                if return_chain:
                    chain.append(x.clone())
        self._generator = None
        self._context = {}
        return SamplerOutput(
            samples=x, chain=torch.stack(chain, dim=0) if return_chain else None
        )


# ── cost.py ───────────────────────────────────────────────────────────────────────────────────
class Cost(ABC):
    """WHAT you tilt toward. `log_h` is the tilt's log-density."""

    @abstractmethod
    def log_h(self, x: "Tensor", t: "Tensor", cond=None) -> "Tensor":
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

    def modify_drift(
        self, drift: "Tensor", x: "Tensor", t: "Tensor", schedule: Schedule, cond=None, context=None
    ) -> "Tensor":
        """GENERAL surface (``surface == "drift"``). Filter / correct the reverse drift b_θ (e.g. a
        CBF QP, an MPPI reweighting). Reads the rate ẋ; not faithfully expressible on x̂₀.

        ``cond`` is the sampler's conditioning spec (steer per-conditioning); ``context`` is the
        sampler's per-sample mutable scratch dict (carry state across reverse steps). Both are optional
        — a controller that needs neither ignores them."""
        raise NotImplementedError

    def modify_x0(
        self, x0_hat: "Tensor", x: "Tensor", t: "Tensor", schedule: Schedule, cond=None, context=None
    ) -> "Tensor":
        """SPECIALIZATION (``surface == "x0"``). Return a shifted clean estimate x̂₀; the sampler
        converts it to an equivalent drift correction via the schedule when needed. ``cond`` /
        ``context`` as in :meth:`modify_drift`."""
        raise NotImplementedError

    def modify_variance(self, sigma, x, t, schedule, cond=None, context=None):
        """Optional: scale/shift the reverse-step noise scale (a sampling-temperature knob). Default:
        identity. Called by stochastic samplers (ancestral / DDPM-style) immediately before building
        the next noise term — a concrete stochastic ``Sampler.step()`` may call
        ``self.control.modify_variance(...)`` directly, guarding for ``self.control is None``.
        Deterministic samplers (DDIM η=0, PF-ODE) have no noise term and never call this. ``context``
        mirrors the other hooks so a history-dependent temperature needs no later signature change."""
        return sigma


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


# ── metric.py ─────────────────────────────────────────────────────────────────────────────────
class Metric(ABC):
    """A swappable evaluation metric. Dependencies are injected at build by ``__init__`` name
    (``environment``/``model``/``method``/``dataset``/``schedule``, Invariant 4), so a metric
    declares only what it needs. Two families by what they consume:
      - sample-driven (MMD/W2/energy/mode-coverage): ``samples`` are the generated RAW-unit points,
        compared against an ``environment.sample(N)`` reference;
      - data-driven (held-out loss/perplexity): ``held_out`` is a NORMALIZED held-out data batch
        run through ``model``+``method``.
    Each call returns a flat ``{name: float}`` dict. A metric that cannot run on the given config
    must RAISE, never silently return ``{}`` (a user-configured metric must not vanish)."""

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None):
        self.environment = environment
        self.model = model
        self.method = method
        self.dataset = dataset
        self.schedule = schedule

    @abstractmethod
    def __call__(self, samples=None, held_out=None) -> dict:
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

# Architecture & Invariants

`genforge` unifies generative paradigms under a single state-evolution abstraction plus an
orthogonal control layer. The pipeline decomposes into small, decoupled components that keep
the math, the training objective, and the steering mechanism independent of one another.

---

## 1. One state-evolution abstraction

### Continuous spaces

For $\mathcal{X} \subseteq \mathbb{R}^d$, generation integrates an SDE:

$$dX_t = b_\theta(X_t, t)\,dt + G(t)\,dW_t, \qquad X_0 \sim p_0, \qquad t \in [0,1]$$

* **Score diffusion (SDE)** — $G(t) \neq 0$; the drift $b_\theta$ is assembled from a learned score $\nabla_{X_t} \log p_t(X_t)$ or an equivalent noise prediction $\epsilon_\theta$.
* **Probability-flow ODE** — $G(t) = 0$ while the marginals $p_t$ are preserved through an adjusted drift; the path is deterministic.
* **Flow matching / OT-CFM** — deterministic ($G(t) = 0$); the network directly regresses a velocity field $b_\theta = v_\theta(X_t, t)$, typically on optimal-transport couplings.
* **Stochastic interpolants** — an explicit stochasticity schedule $G(t) = \sqrt{2\,\varepsilon(t)}$; $\varepsilon$ can be changed at inference time without retraining, since the marginals do not depend on it.

### Discrete spaces (D3PM and jump processes)

For categorical data (tokens, discrete structures), the vector field is replaced by a
**Markov jump process** over $K$ categories. The forward process corrupts data by
transitioning between states according to a time-dependent transition matrix $Q_t$:

$$q(X_t \mid X_{t-1}) = X_{t-1} Q_t \quad \implies \quad q(X_t \mid X_0) = X_0 \bar{Q}_t, \qquad \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

* **Corruption types** — uniform (jump to any token), absorbing (replace with a `[MASK]` token), or structured matrices (local jumps).
* **Reverse step** — the network predicts the clean categorical distribution $\hat{p}_\theta(X_0 \mid X_t)$; the sampler applies Bayes' rule to get the posterior transition and samples the jump:

$$q(X_{t-1} \mid X_t, X_0) = \frac{q(X_t \mid X_{t-1}, X_0) \, q(X_{t-1} \mid X_0)}{q(X_t \mid X_0)}$$

Discrete models run inside the exact same sampling loop as continuous ones.

---

## 2. Three primitives

A generative paradigm joins the framework by implementing exactly three operations; nothing
else in the stack needs to know which paradigm it is:

1. **forward** — `space.forward_sample(x0, t, schedule)` corrupts clean data into $x_t \sim q(x_t \mid x_0)$ *(training)*.
2. **reverse step** — `sampler.step(x, t, s)` takes one reverse increment from time $t$ to $s$ *(sampling)*.
3. **objective** — `method.loss(model, x0)` computes the training loss.

---

## 3. Composition & pipeline

A central registry maps `(category, name)` to a class. A config-driven builder constructs each
component and **injects the already-built ones it names** into its constructor (dependency by name,
not a linear chain). Solid = required dependency, dashed = optional; dashed nodes are optional to
define. Everything terminates at the `Runner`, which owns the lifecycle.

<div class="gf-fig">
<div class="gf-summary">
<div class="gf-row"><span class="gf-tag">Compulsory</span><span class="gf-chip gf-req">space</span><span class="gf-chip gf-req">schedule</span><span class="gf-chip gf-req">model</span><span class="gf-chip gf-req">method</span><span class="gf-chip gf-req">sampler</span><span class="gf-chip gf-req">dataset</span><span class="gf-chip gf-req">runner</span></div>
<div class="gf-row"><span class="gf-tag">Optional</span><span class="gf-chip gf-opt">criterion</span><span class="gf-chip gf-opt">cost</span><span class="gf-chip gf-opt">control</span><span class="gf-chip gf-opt">environment</span><span class="gf-chip gf-opt">preprocessor</span><span class="gf-chip gf-opt">visualizer</span><span class="gf-chip gf-opt">metric</span></div>
</div>
<div class="gf-assembly">
<div class="gf-arow"><span class="gf-lhs">method</span><span class="gf-eq">=</span><span class="gf-chips"><span class="gf-chip gf-req">schedule</span><span class="gf-chip gf-req">space</span><span class="gf-chip gf-opt">criterion</span></span></div>
<div class="gf-arow"><span class="gf-lhs">control</span><span class="gf-eq">=</span><span class="gf-chips"><span class="gf-chip gf-opt">cost</span></span></div>
<div class="gf-arow"><span class="gf-lhs">sampler</span><span class="gf-eq">=</span><span class="gf-chips"><span class="gf-chip gf-req">model</span><span class="gf-chip gf-req">schedule</span><span class="gf-chip gf-req">space</span><span class="gf-chip gf-opt">control</span></span></div>
<div class="gf-arow"><span class="gf-lhs">data&nbsp;lane</span><span class="gf-eq">·</span><span class="gf-chips"><span class="gf-chip gf-opt">environment</span><span class="gf-arrowg">→</span><span class="gf-chip gf-req">dataset</span><span class="gf-arrowg">→</span><span class="gf-chip gf-opt">preprocessor</span></span></div>
<div class="gf-arow"><span class="gf-lhs">runner</span><span class="gf-eq">=</span><span class="gf-chips"><span class="gf-chip gf-req">model</span><span class="gf-chip gf-req">method</span><span class="gf-chip gf-req">sampler</span><span class="gf-chip gf-req">space</span><span class="gf-chip gf-req">schedule</span><span class="gf-chip gf-req">dataset</span><span class="gf-chip gf-opt">environment</span><span class="gf-chip gf-opt">preprocessor</span><span class="gf-chip gf-opt">visualizer</span><span class="gf-chip gf-opt">metric</span></span></div>
</div>
<div class="gf-legend"><span><span class="gf-chip gf-req">required</span> must be defined</span><span><span class="gf-chip gf-opt">optional</span> feature off if omitted</span></div>
</div>

The **`Method`** is assembled from `schedule` + `space` (+ optional `criterion`); the **`Sampler`**
from `model` + `schedule` + `space` (+ optional `control`, which itself takes an optional `cost`).
The **`Runner`** then receives `method`, `sampler`, `dataset` (required) plus any optional
`environment` / `preprocessor` / `visualizer` / `metric`. Omitting a dashed node just turns that
feature off; omitting a solid one is a build error.

| Component | Responsibility |
| --- | --- |
| **`Space`** | The state space and forward corruption kernel. The **sole** component that distinguishes continuous vectors from discrete tokens. |
| **`Schedule`** | The time-noise mapping on $t \in [0, 1]$; owns every algebraic conversion between noise ($\epsilon$), clean data ($x_0$), score ($\nabla \log p$), and velocity ($v$). |
| **`Criterion`** | The swappable per-sample regression penalty a `Method` reduces with (MSE, Huber). Optional; defaults to MSE. |
| **`Model`** | The network (MLP, temporal UNet, transformer); declares its prediction target via an `output_type` attribute. |
| **`Method`** | The training objective. |
| **`Sampler`** | The reverse-path integration loop and its numerical solver (Euler–Maruyama, Heun, DDIM, τ-leaping). |
| **`Preprocessor`** | The normalization membrane between raw data units and the normalized coordinates used internally. |
| **`Cost` / `Control`** | The steering layer: what to steer toward, and how the correction is computed. |
| **`Metric`** | A swappable evaluation metric (distribution distance, held-out likelihood, coverage); results are persisted to `metrics.json`. Runs a list via `MetricSet`. |
| **`Runner`** | The experiment lifecycle: training, sampling, logging, checkpointing, evaluation. |

### Required vs. optional components

Two layers decide what a config **must** define:

1. **The builder** hard-requires exactly one leaf — **`Runner`** (it returns a ready runner, so it raises if none is configured), plus at least one component overall.
2. **Constructor injection** does the rest: the builder passes each built component into any later component whose `__init__` declares a parameter of the same name. A category is therefore *required* whenever some built component names it as a **no-default** parameter (omitting it raises `TypeError`); a `= None` default makes it optional.

The `Runner` drives the compulsory set — its required constructor parameters transitively pull in everything needed to train and sample.

| Component | Required? | Needed by | If omitted |
| --- | --- | --- | --- |
| **`Runner`** | **Compulsory** | *(the builder returns it)* | build error |
| **`Space`** | **Compulsory** | `Method`, `Sampler`, `Runner` | build error |
| **`Schedule`** | **Compulsory** | `Method`, `Sampler`, `Runner` | build error |
| **`Model`** | **Compulsory** | `Sampler`, `Runner` | build error |
| **`Method`** | **Compulsory** | `Runner` | build error |
| **`Sampler`** | **Compulsory** † | `Runner` | build error |
| **`Dataset`** | **Compulsory** | `Runner` | build error |
| **`Criterion`** | Optional | `Method` | `Method` uses its built-in loss (MSE) |
| **`Cost`** | Optional ‡ | `Control` | controller runs unguided |
| **`Control`** | Optional | `Sampler` | plain, unguided sampling |
| **`Environment`** | Optional § | `Runner` | no eval reference draw / no rollout |
| **`Preprocessor`** | Optional | `Runner` | train + sample in **raw** units (no membrane) |
| **`Visualizer`** | Optional | `Runner` | nothing rendered |
| **`Metric`** | Optional | `Runner` | falls back to the environment's own metric |

† `Sampler` is compulsory only because every shipped runner extends the training runner, which lists it as a required argument. A hypothetical train-only runner could relax it.

‡ `Cost` is optional to *build*, but a specific controller may require one (gradient guidance needs a cost's $\log h$; a CBF needs a barrier).

§ `Environment` is optional to *train*, but required for evaluation metrics that draw a reference (MMD, Wasserstein-2, mode coverage all call `environment.sample`) and for rollout runners (planning, policy). `Environment` and `Dataset` have **no config group** — set them inline in the experiment (`name` + `params`), and load their implementations via the experiment's `plugins:` field.

!!! note "Minimal runnable config"
    The seven compulsory leaves are enough to build and train:

    ```yaml
    space:    {name: euclidean,    params: {dim: 2}}
    schedule: {name: vp_linear,    params: {}}
    model:    {name: mlp,          params: {dim: 2, output_type: eps}}
    method:   {name: ddpm,         params: {}}
    sampler:  {name: ddpm,         params: {}}
    dataset:  {name: distribution, params: {n_samples: 4096}}
    runner:   {name: training,     params: {steps: 1000}}
    ```

    Every optional leaf you add (a preprocessor membrane, a cost + controller for guided sampling, a metric, a visualizer) turns on a feature; omitting it means that feature is simply off — never a crash.

### Runtime pipeline

Composition is *build-time*; at *run-time* those components drive two flows that cross the
normalization membrane, each expressed through the **three primitives** (`forward`, `reverse`,
`objective`):

<div class="gf-fig">
<div class="gf-grid2 gf-heads"><div class="gf-h"><span class="gf-dot"></span>Training</div><div class="gf-h"><span class="gf-dot"></span>Sampling</div></div>
<div class="gf-grid2">
<div class="gf-col"><div class="gf-node gf-raw"><span class="gf-k">raw units</span><code>dataset.batch → x_0</code></div><div class="gf-cross">↓ preprocessor.transform</div></div>
<div class="gf-empty">sampling begins inside the membrane →</div>
</div>
<div class="gf-mem"><span class="gf-mlabel">normalized coordinates — inside the membrane</span>
<div class="gf-grid2">
<div class="gf-col"><div class="gf-node gf-p"><span class="gf-k">forward</span><code>space.forward_sample(x_0, t) → x_t</code></div><div class="gf-conn">↓</div><div class="gf-node gf-p"><span class="gf-k">objective</span><code>method.loss(model, x_0)</code></div></div>
<div class="gf-col"><div class="gf-node"><span class="gf-k">prior</span><code>space.prior_sample → x_T</code></div><div class="gf-conn">↓</div><div class="gf-node gf-p"><span class="gf-k">reverse · step loop</span><code>model → schedule convert → control tilt → pin</code></div><div class="gf-conn">↓</div><div class="gf-node"><span class="gf-k">clean estimate</span><code>x_0 (normalized)</code></div></div>
</div>
</div>
<div class="gf-grid2">
<div class="gf-col"><div class="gf-cross">↓ backward</div><div class="gf-node"><span class="gf-k">optimize</span><code>optimizer step + EMA</code></div></div>
<div class="gf-col"><div class="gf-cross">↓ preprocessor.inverse</div><div class="gf-node gf-raw"><span class="gf-k">raw units</span><code>x_0</code></div><div class="gf-conn">↓</div><div class="gf-node"><span class="gf-k">score + persist</span><code>Metric · Visualizer → samples.npz / metrics.json</code></div></div>
</div>
</div>

**Training** (left branch) corrupts a data batch with `forward` and minimizes the `objective`;
**sampling** (right branch) starts from the prior and walks the `reverse` step loop — where the
control layer tilts the path and boundary conditions are re-pinned — then inverts the membrane and
scores/persists. Raw units exist only *outside* the membrane; everything in between is normalized.

---

## 4. The normalization membrane

Networks train best on well-conditioned inputs, so `genforge` enforces a strict normalization
boundary, owned by the `Preprocessor`:

* Statistics are fitted once on the training data and travel in the checkpoint.
* Data is normalized on the way in; generated samples are mapped back to raw units on the way out.
* Everything in between — `Model`, `Method`, `Sampler`, `Control` — operates only on normalized tensors.
* The membrane rescales **only the generated quantity** (states, actions, trajectories). Conditioning inputs (images, proprioception) pass through untouched; normalizing them is the model's job (e.g. a vision encoder inside the model).

---

## 5. Control layer: tilting the path measure

Conditioning, safety, guidance, and reward maximization are all framed as a change of the
base path measure: $Q \propto \exp(\log h(X)) \cdot P$. The layer separates *what* is wanted
from *how* the trajectory is modified:

* **`Cost`** — returns the potential $\log h(X)$: a likelihood, a reward, a constraint, a safety barrier.
* **`Controller`** — computes the correction, on one of two surfaces:
    * **`drift`** — perturbs the reverse drift directly. This is the general surface, and the required one whenever the tilt reads the rate $\dot{x}$ — e.g. control barrier function (CBF) safety filters.
    * **`x0`** — shifts the model's clean-data estimate $\hat{x}_0$. A convenient specialization for projection and guidance; the sampler converts the shift into an equivalent drift correction via the schedule.

An `x0` correction always converts to a drift correction; the reverse does not hold, because
$\hat{x}_0$ has discarded the rate. That is why both surfaces exist.

---

## 6. Invariants

1. **The continuous/discrete distinction lives only in `Space` and `Schedule`.** No other component branches on it.
2. **Everything inside the membrane is normalized.** `Model`, `Method`, `Sampler`, and `Control` see only normalized tensors; raw units exist only at the data input and the sample output.
3. **Output-type conversion lives in `Schedule`, nowhere else.** Converting between noise, clean-data, score, and velocity predictions is pure schedule math; samplers and methods stay agnostic to the model's `output_type`.
4. **Dependencies are injected at construction.** The builder wires components once; hot-path signatures stay clean.
5. **Checkpoints are self-contained.** One `.pt` holds weights, EMA shadow, preprocessor statistics, the resolved config, optimizer/RNG state, and provenance — enough to sample from the file alone and to resume training exactly.
6. **Control modifies reverse dynamics only.** The base model stays frozen at sampling time; steering never mutates its weights.
7. **Components register themselves** via `@register(category, name)` and are discovered by the builder — adding one requires no other wiring.
8. **Constraints are authored in real units** and mapped into normalized coordinates by the framework before enforcement.

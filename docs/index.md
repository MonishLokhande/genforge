# genforge

A unified PyTorch framework for process-based generative modeling — score/diffusion SDEs,
probability-flow ODEs, flow matching / OT-CFM, stochastic interpolants, and discrete
(D3PM-style) diffusion — with a clean **control layer** for conditioning, guidance,
constraints, rewards, and amortized control.

---

## The unifying idea

`genforge` is built on one principle: **a generative model defines a state-evolution process
that carries a simple prior to data, and every way of steering that model — conditioning,
guidance, constraints, planning — is a control-theoretic perturbation of the same process.**

Rather than treating diffusion, flow matching, and discrete token processes as separate
paradigms, `genforge` expresses them all as forward and reverse SDEs, ODEs, and Markov jump
processes.

### Continuous dynamics

In a continuous state space $\mathcal{X} \subseteq \mathbb{R}^d$, generation is a stochastic
or deterministic process on $t \in [0, 1]$:

$$dX_t = b_\theta(X_t, t)\,dt + G(t)\,dW_t, \qquad X_0 \sim p_0, \quad X_1 \sim p_{\text{data}}$$

Score diffusion, probability-flow ODEs, flow matching, and stochastic interpolants are all
this one SDE with different choices of drift $b_\theta$, diffusion $G(t)$, and boundary
conditions.

??? info "How each paradigm instantiates the SDE"

    $b_\theta$ is the learned drift or velocity field, $G(t)$ the diffusion coefficient, and
    $W_t$ standard Brownian motion.

    | Paradigm | Dynamics | Drift $b_\theta$ |
    | :--- | :--- | :--- |
    | **Score diffusion / DDPM** | Stochastic | Built from a learned score $\nabla_x \log p_t(x)$ or noise predictor $\epsilon_\theta(X_t, t)$ via Tweedie's formula. |
    | **Probability-flow ODE** | Deterministic | Noise turned off while the marginals $p_t(x)$ are preserved through the adjusted drift $\tilde{b} = b_\theta - \frac{1}{2}G(t)G(t)^T \nabla_x \log p_t(x)$. |
    | **Flow matching / OT-CFM** | Deterministic | A velocity field regressed directly onto interpolation paths, typically optimal-transport couplings (OT-CFM). |
    | **Stochastic interpolants** | Either | Explicit paths between arbitrary boundary distributions $p_0$ and $p_1$ with paired velocity and score fields; generalizes both diffusion and flow matching. |

### Discrete dynamics

For categorical data (tokens, discrete structures), the same role is played by a **Markov
jump process** over $K$ categories: corruption follows a transition matrix $Q_t$ instead of
Gaussian noise, and the reverse step samples jumps instead of integrating a drift
(D3PM-style diffusion).

??? info "The discrete forward and reverse process"

    The forward process corrupts data by transitioning between states according to a
    time-dependent transition matrix $Q_t$:

    $$q(X_t \mid X_{t-1}) = X_{t-1} Q_t \quad \implies \quad q(X_t \mid X_0) = X_0 \bar{Q}_t, \qquad \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

    Corruption can be **uniform** (jump to any token), **absorbing** (replace with a `[MASK]`
    token), or **structured** (local jumps). For the reverse step, the network predicts the
    clean categorical distribution $\hat{p}_\theta(X_0 \mid X_t)$; the sampler applies Bayes'
    rule to get the posterior transition and samples the jump:

    $$q(X_{t-1} \mid X_t, X_0) = \frac{q(X_t \mid X_{t-1}, X_0) \, q(X_{t-1} \mid X_0)}{q(X_t \mid X_0)}$$

    Discrete models run inside the exact same sampling loop as continuous ones.

The framework treats these as variations of one abstraction. Corruption schedules, reverse
steps, and training objectives are isolated components, so nothing downstream needs to know
which paradigm is running.

---

## The control layer

Once a base model is trained, downstream tasks — conditional generation, constraint
satisfaction, reward maximization, safety filtering — are all the same operation: sampling
from a *tilted* version of the base process. Let $P$ be the path measure (the distribution
over trajectories) induced by the unconditional model. The steered target $Q$ is a change of
measure

$$\frac{dQ}{dP}(X) = \frac{1}{Z}\, h(X)$$

where $h(X) \geq 0$ encodes a likelihood, reward, constraint, or safety barrier, and
$Z = \mathbb{E}_P[h(X)]$ normalizes. By Girsanov's theorem and Doob's $h$-transform, the tilt
appears as an extra steering term on the drift:

$$dX_t = \left( b_\theta(X_t, t) + G(t)G(t)^T \nabla_{X_t} \log h_t(X_t) \right) dt + G(t)\,dW_t$$

where $h_t(X_t) = \mathbb{E}_P [h(X) \mid X_t]$. `genforge` splits this into two decoupled
pieces:

* **`Cost`** — *what* you steer toward: evaluates the trajectory or terminal state and returns the potential $\log h$.
* **`Controller`** — *how* the tilt is approximated: computes the correction ($\nabla \log h_t$ or a substitute for it) and injects it into the sampler at runtime.

One `Sampler` implementation then supports classifier guidance, value-function control,
projection onto constraint sets, and control barrier functions without modifying the base
model.

---

## Architecture

The framework is a graph of orthogonal components wired by **dependency injection** — the builder
constructs each and passes the already-built ones it names into its constructor. A config fully
determines the graph. Solid = required dependency, dashed = optional (dashed nodes are optional to
define); everything terminates at the `Runner`.

<div class="gf-fig">
<div class="gf-summary">
<div class="gf-row"><span class="gf-tag">Compulsory</span><span class="gf-chip gf-req">space</span><span class="gf-chip gf-req">schedule</span><span class="gf-chip gf-req">model</span><span class="gf-chip gf-req">method</span><span class="gf-chip gf-req">sampler</span><span class="gf-chip gf-req">dataset</span><span class="gf-chip gf-req">runner</span></div>
<div class="gf-row"><span class="gf-tag">Optional</span><span class="gf-chip gf-opt">criterion</span><span class="gf-chip gf-opt">cost</span><span class="gf-chip gf-opt">control</span><span class="gf-chip gf-opt">environment</span><span class="gf-chip gf-opt">preprocessor</span><span class="gf-chip gf-opt">visualizer</span><span class="gf-chip gf-opt">metric</span></div>
</div>
</div>

See [Architecture](architecture.md) for the runtime pipeline and the required-vs-optional table.

| Component | Responsibility |
| --- | --- |
| `Space` | The state space and forward corruption kernel (`Euclidean`, `Discrete`; `Product` planned). The only home of the continuous-vs-discrete distinction. |
| `Schedule` | The noise/time schedule on $t \in [0,1]$; owns every algebraic conversion between noise ($\epsilon$), clean data ($x_0$), score, and velocity. |
| `Criterion` | The swappable per-sample regression penalty a `Method` reduces with (MSE, Huber). Optional; defaults to MSE. |
| `Model` | The neural network (MLP, temporal UNet, transformer). Declares its prediction target via an `output_type` attribute. |
| `Method` | The training objective (variational bound, score matching, flow-matching MSE, ...). |
| `Sampler` | Integrates the reverse path (Euler–Maruyama, Heun, DDIM, τ-leaping). |
| `Cost` | The steering target: potentials, constraints, barriers. |
| `Control` | The steering mechanism: guidance gradients, projections, safety filters. |
| `Preprocessor` | The normalization membrane: components inside it see only normalized tensors; raw units exist only at the input/output boundary. |
| `Metric` | A swappable evaluation metric (distribution distance, held-out likelihood, coverage); results persist to `metrics.json`. Optional. |
| `Runner` | The experiment lifecycle: train, sample, evaluate, checkpoint. |

---

## Installation

Install the core framework from PyPI:

```bash
pip install genforge          # or: uv add genforge
```

> **Note:** the PyPI wheel ships the framework only. The bundled `experiment/` recipes and
> `envs/` data-source plugins live in the source repository — running the bundled experiments
> requires a clone.

For development or the bundled experiments:

```bash
git clone https://github.com/MonishLokhande/genforge
cd genforge

uv sync                   # core (light: 2-D synthetic distributions)
uv sync --extra flow      # + optimal transport / OT-CFM (scipy)
uv sync --extra text      # + BPE tokenization and streamed text datasets
uv sync --extra logging   # + wandb logging and tqdm progress bars
```

Robotics environments are a dependency group (one dependency installs from git):

```bash
uv sync --group robotics
```

---

## Quickstart

Verify the installation and inspect the component registry:

```bash
uv run forge list
```

Train, sample, and evaluate a baseline diffusion model:

```bash
uv run forge train  experiment=distributions/ddpm/base
uv run forge sample experiment=distributions/ddpm/base
uv run forge eval   experiment=distributions/ddpm/base   # score + persist samples.npz / metrics.json
```

### Sampling from a checkpoint alone

Every checkpoint is self-contained — weights, EMA shadow, preprocessor statistics, resolved
config, and provenance. You can sample from a `.pt` file without restating any configuration:

```bash
uv run forge sample checkpoint=<path>.pt
```

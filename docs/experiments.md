# Experiments

Workloads are declared as configs under the `experiment/` tree and selected with
`experiment=<family>/<variant>/<method>`. Each family has a **base** recipe holding the
shared setup; leaf files hold only the deltas they override.

```bash
uv run forge list                                       # registered components
uv run forge train  experiment=distributions/ddpm/base  # train
uv run forge sample experiment=distributions/ddpm/base  # sample from the trained model
uv run forge sample checkpoint=<path>.pt                # sample from a checkpoint alone
```

---

## 1. Experiment families

### Continuous 2-D distributions (`distributions/*`)

A fast synthetic suite for validating continuous dynamics, schedules, samplers, and the
control layer.

| Experiment | Paradigm | What it exercises |
| --- | --- | --- |
| `distributions/ddpm/base` | Score diffusion (SDE) | DDPM on a bimodal Gaussian mixture. |
| `distributions/ddpm/standardize` | Score diffusion (SDE) | The same recipe run through the normalization membrane. |
| `distributions/ddpm/huber` | Score diffusion (SDE) | The DDPM recipe with the loss `criterion` swapped MSE → Huber (smooth $L_1$); everything else inherited. |
| `distributions/flow/base` | Flow matching | A deterministic velocity field trained on linear (rectified-flow) paths. |
| `distributions/interpolant/base` | Stochastic interpolants | Explicit interpolation paths; setting the inference-time coefficient $\varepsilon(t) = 0$ collapses the SDE into a probability-flow ODE. |
| `distributions/ddim/base` | DDIM | Deterministic non-Markovian reverse discretization. |
| `distributions/ddpm/halfspace_{project,guide,cbf}` | Control layer | Constraint handling three ways: hard **projection**, gradient **guidance**, and a **CBF** safety filter. |
| `distributions/value/{values,guided}` | Amortized control | Trains a separate value network over path costs, then uses its gradient for value-guided sampling. |

### Discrete toy (`discrete/d3pm/base`)

A minimal categorical setup: absorbing-state (mask) discrete diffusion on a skewed toy
target.

### Discrete diffusion language models (`text/*`)

Categorical jump processes over language tokens, at two scales. All variants share a
transformer backbone, an absorbing noise schedule, and the `envs.text` plugin.

| Method | Objective | Sampler |
| --- | --- | --- |
| `d3pm` | $x_0$-prediction cross-entropy | $\tau$-leaping |
| `mdlm` | Masked, rate-weighted continuous-time NELBO | $\tau$-leaping |
| `sedd` | Denoising score entropy | Score-entropy jump sampler |

The two scales:

* **Char-level** (`text/char/{method}/{base,small}`) — the `char_text` environment; vocabulary $K = 32$, fast enough for CPU.
* **TinyStories BPE** (`text/tinystories/{method}/small`) — the `tinystories` streaming environment; full GPT-2 BPE vocabulary ($K = 50{,}258$). *Needs `--extra text` and a GPU.*

Two extra char-level variants:

* `text/char/d3pm/finetune` — starts from a trained `small` checkpoint and fine-tunes with LoRA adapters, base weights frozen.
* `text/char/mdlm/timeless` — replaces time embeddings with a time-independent RoPE backbone; skipping per-step time reconditioning makes sampling roughly 2× faster.

```bash
uv run forge train experiment=text/char/d3pm/small                      # char-level LM
uv run --extra text forge train experiment=text/tinystories/d3pm/small  # same method, real BPE
```

### Goal-conditioned planning (`trajectory/plan/base`)

A continuous trajectory diffuser for sequential planning.

* **Flat-tensor windowing** — trajectory windows are sliced on the fly from flat state arrays instead of materialized up front (roughly 90× less memory).
* **Endpoint pinning** — the sampler re-imposes the start and goal states (the first and last timesteps of the planned trajectory) at every reverse step; the pins are pushed through the min-max membrane so the endpoints are exact in raw units.

### Robotics (`robotics/*`)

*Needs `uv sync --group robotics`.*

* **Tier A — trajectory planning**: `robotics/maze2d/{umaze,medium,large}/ddpm` and `robotics/locomotion/{halfcheetah,hopper,walker2d}/ddpm`. Whole-trajectory diffusers over offline datasets (D4RL/Minari), reusing the flat-tensor windowing and endpoint pinning from `trajectory/plan`.
* **Tier B — closed-loop policies**: `robotics/robomimic/{can,lift}/ddpm`, `robotics/pusht/ddpm`, and `robotics/aloha/ddpm`. High-frequency action-sequence diffusers conditioned on observations. Observation conditioning is normalized by an in-model `ObsNormalizer`, not the framework membrane — the membrane touches only the generated quantity.

---

## 2. Logging

Logging is off by default and its dependencies are optional:

```bash
uv sync --extra logging   # wandb + tqdm
```

* **Weights & Biases** — enable with `runner.params.log.wandb=true` or the environment variable `FORGE_WANDB=1`. Project, run name, and mode are set via `runner.params.log.project` and `runner.params.log.mode=online|offline|disabled`. Logs step loss and learning rate every `log_every` steps.
* **Progress bars** — `runner.params.log.progress=true` toggles tqdm.

If the `logging` extra isn't installed, or the terminal is not a TTY (CI, shell pipes), the
logger degrades to a silent no-op — nothing errors, and `wandb` is never imported unless it
is actually used.

```bash
uv run --extra logging forge train experiment=distributions/ddpm/base \
  runner.params.log.wandb=true \
  runner.params.log.project=genforge_benchmarks \
  runner.params.log.progress=true
```

---

## 3. Config layout

The configuration tree separates framework defaults from experiment recipes:

```text
src/forge/configs/                # framework component defaults
├── space/
├── schedule/
├── model/                        # MLP, temporal UNet, transformer
├── method/
├── sampler/
├── cost/
├── control/
└── preprocessor/                 # standardize / minmax

experiment/                       # the experiment tree
└── <family>/
    └── <variant>/
        └── <method>.yaml         # leaf config; environment declared inline
```

* **Inline environments** — a leaf declares its `environment`/`dataset` inline (e.g. `environment: {name: tinystories, params: {batch_size: 64}}`) and loads the plugin package via its `plugins:` field.
* **Defaults** — a leaf only states overrides; anything omitted falls back to the defaults in `src/forge/configs/`.
* **Experiment root** — the tree is located via the `GENFORGE_EXP_ROOT` environment variable; if unset, the current working directory is used.

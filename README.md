# genforge

A unified, PyTorch framework for generative modeling — score/diffusion SDEs, probability-flow
ODEs, flow matching / OT-CFM, stochastic interpolants, and discrete (D3PM-style) diffusion — with a
clean **control layer** for conditioning, guidance, constraints, and amortized control.

The idea: a generative model is one process that turns a simple starting distribution (usually noise)
into data, and every way of steering it —
conditioning, guidance, constraints, planning — is the **same move**: reweight that process so the
outcomes you want become more likely. The process, the steering, and how the steering is approximated
are separate, swappable parts. A full documentation site (architecture, experiments, extending) is coming soon.

## Install

**Use it as a library:**

```bash
pip install genforge          # or: uv add genforge
```

The bundled `experiment/` tree used in the Quickstart below ships with the **source clone**, not
the PyPI wheel — running `forge train experiment=...` requires `git clone` + `uv sync`, not just
`pip install genforge`.

**Develop / run the bundled experiments:**

```bash
git clone https://github.com/MonishLokhande/genforge
cd genforge
uv sync                   # core (light: 2-D distributions)
uv sync --extra flow      # + OT-CFM (scipy)
uv sync --extra text      # + real BPE / streamed corpora (tiktoken, datasets)
uv sync --extra logging   # + experiment logging (wandb) + progress bars (tqdm)
```

Robotics adapters are a **dependency group**, not an extra (one dependency installs from git):
`uv sync --group robotics` (mujoco, robomimic, gym-pusht/aloha, minari).

## Quickstart

```bash
uv run forge list                                       # registered components
uv run forge train  experiment=distributions/ddpm/base
uv run forge sample experiment=distributions/ddpm/base  # or: sample checkpoint=<path>.pt
```

## Layout

Every component registers via `@register(category, name)` and is wired by a config-driven builder in
dependency order — adding one is a single decorated class plus a config leaf, no other wiring.

- **`src/forge/`** — the framework only: `core` (registry · builder · interfaces · protocols ·
  plugins), `spaces`, `schedules`, `models`, `methods`, `samplers`, `costs`, `control`,
  `preprocessing`, `runners`. Protocols, ABCs, and generic utilities — never concrete env code.
- **`envs/`** — concrete, swappable **data-source plugins** (environment + dataset + processor per
  package). An experiment loads them via its `plugins:` field; they are not baked into the core.
  Contracts: [`core/protocols.py`](src/forge/core/protocols.py).
- **`experiment/`** — Hydra base+delta bundles, selected with `experiment=<family>/<variant>/<method>`.

## Experiments

Selected with `experiment=<family>/<variant>/<method>`.

| Family | What |
|---|---|
| `distributions/*` | Continuous 2-D — DDPM, flow matching, stochastic-interpolant SDE, DDIM; control via projection / guidance / CBF; value guidance. |
| `discrete/d3pm/base` | Discrete (absorbing) diffusion on a toy categorical target. |
| `text/char/*` | Discrete diffusion LM, char-level — `d3pm` / `mdlm` / `sedd`. |
| `text/tinystories/*` | The **same** methods at real GPT-2 BPE (vocab 50258, needs `--extra text`). |
| `trajectory/plan/base` | Goal-conditioned trajectory planning (flat-tensor windowing, endpoint-pinned). |
| `robotics/*` | Offline-RL trajectory planning (maze2d, locomotion) and closed-loop diffusion policies (robomimic, pusht, aloha); needs `--group robotics`. |

`text/char/*` and `text/tinystories/*` are two variants of one **`text` family** (a single env plugin
`envs.text` registers both) — the same absorbing + transformer + `{d3pm,mdlm,sedd}` rig; only the
tokenizer (char vs. real BPE) and scale differ.

```bash
uv run forge train experiment=text/char/d3pm/small                     # char-level LM
uv run --extra text forge train experiment=text/tinystories/d3pm/small # same method, real BPE
```

## Documentation

A full documentation site — installation, architecture, experiments, and extending — is coming soon.
Until then, this README plus the inline docstrings across `src/forge/` are the reference.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use genforge in your work, please cite it — citation metadata is in [CITATION.cff](CITATION.cff).

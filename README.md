# genforge

`genforge` is a PyTorch framework for process-based generative modeling. It gives score/diffusion
SDEs, probability-flow ODEs, flow matching / OT-CFM, stochastic interpolants, and discrete
D3PM-style diffusion the same component graph, then adds a control layer for conditioning,
guidance, constraints, rewards, planning, and amortized control.

The package name is `genforge`; the Python import and CLI are `forge`.

## Why It Exists

Most generative-modeling codebases split by paradigm: one stack for diffusion, another for flow
matching, another for discrete token diffusion, another for guided planning. `genforge` keeps the
split in the right place:

- `Space` and `Schedule` define the state space and corruption/time algebra.
- `Method` defines the training objective.
- `Sampler` defines one reverse-time integration or jump step.
- `Cost` and `Control` steer a trained process without changing the base model.
- `Runner` owns training, sampling, evaluation, logging, checkpointing, and resume.

Every concrete component registers itself with `@register(category, name)`. A Hydra config names
the leaves, and the builder injects dependencies by constructor name. Adding a component is a class
plus, when useful, a config leaf.

## Install

Use the published package when you want the framework contracts and the built-in reference path:

```bash
pip install genforge          # or: uv add genforge
```

The wheel ships `src/forge` and one runnable 2-D DDPM stack:
`euclidean` space, VP schedules, MLP model, DDPM method/sampler, preprocessors, metrics composer,
visualizers, and the base training runner.

Clone the repository when you want the bundled experiment tree, source examples, and data-source
plugins:

```bash
git clone https://github.com/MonishLokhande/genforge
cd genforge
./install.sh
```

`./install.sh` installs `uv` if needed, then runs `uv sync`. Extra arguments pass through:

```bash
./install.sh --extra flow --extra logging
./install.sh --group robotics
```

Optional extras and groups:

| Option | Adds | Use For |
| --- | --- | --- |
| `--extra flow` | `scipy` | OT-CFM coupling and exact W2 metric |
| `--extra text` | `tiktoken`, `datasets` | GPT-2 BPE tokenization and streamed text corpora |
| `--extra logging` | `wandb`, `tqdm` | Weights & Biases logging and progress bars |
| `--group docs` | `mkdocs-material` | Local documentation site |
| `--group robotics` | MuJoCo, robosuite, robomimic, Minari, PushT/Aloha deps | Robotics adapters and policies |

Robotics is a dependency group, not a PyPI extra, because one dependency is installed from git.

## Quickstart

List registered components:

```bash
uv run forge list
```

Run the fast 2-D DDPM recipe from the source checkout:

```bash
uv run forge train  experiment=distributions/ddpm/base
uv run forge sample experiment=distributions/ddpm/base
uv run forge eval   experiment=distributions/ddpm/base
```

For a quick smoke run, override the training length without editing YAML:

```bash
uv run forge train experiment=distributions/ddpm/base runner.params.steps=2 runner.params.log_every=0
```

Checkpoints are self-contained. They include weights, EMA state, preprocessor statistics, resolved
config, optimizer/RNG state, and provenance:

```bash
uv run forge sample checkpoint=<path>.pt
uv run forge eval   checkpoint=<path>.pt
```

Offline evaluation can rescore saved samples without regenerating:

```bash
uv run forge eval samples=<path>/samples.npz checkpoint=<path>.pt
```

## Repository Layout

```text
src/forge/       framework package plus the built-in 2-D DDPM reference path
examples/        bundled plugin catalog for extra methods, samplers, models, costs, controls, metrics, runners
envs/            data-source plugins: distributions, discrete toys, text, trajectories, robotics
experiment/      Hydra experiment recipes selected with experiment=<family>/<variant>/<method>
docs/            MkDocs documentation
tests/           unit and integration tests
scripts/         maintainer utilities
```

The installed wheel does not include `examples/`, `envs/`, or `experiment/`. Bundled source
experiments load those modules through their `plugins:` field, usually:

```yaml
plugins:
  - examples
  - envs.distributions
```

Downstream projects can use the same mechanism with their own importable modules, or publish a
package that advertises a `forge.plugins` entry point.

## Experiment Families

Experiments live under `experiment/` and compose base recipes plus small leaf overrides.

| Family | Examples | What It Exercises |
| --- | --- | --- |
| Continuous 2-D | `distributions/ddpm/base`, `distributions/flow/base`, `distributions/interpolant/base` | DDPM, flow matching, stochastic interpolants, DDIM, metrics, preprocessors |
| Control | `distributions/ddpm/halfspace_project`, `halfspace_guide`, `halfspace_cbf` | Projection, gradient guidance, and control barrier functions |
| Value guidance | `distributions/value/values`, `distributions/value/guided` | Amortized control from a learned value network |
| Discrete toy | `discrete/d3pm/base` | Absorbing-state categorical diffusion |
| Text | `text/char/{d3pm,mdlm,sedd}/*`, `text/tinystories/{d3pm,mdlm,sedd}/small` | Discrete diffusion language models at char and GPT-2 BPE scale |
| Trajectory planning | `trajectory/plan/base` | Goal-conditioned trajectory diffusion with endpoint pinning |
| Robotics | `robotics/maze2d/*`, `robotics/locomotion/*`, `robotics/robomimic/*`, `robotics/pusht/ddpm`, `robotics/aloha/ddpm` | Offline-RL planning and closed-loop diffusion policies |

Text BPE recipes need `--extra text`. Robotics recipes need `--group robotics`; on headless servers
set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`.

## Configuration Pattern

Each selectable category has config leaves under `src/forge/configs/<category>/`, while an
experiment leaf declares the environment and dataset inline. Use Hydra overrides to change runs:

```bash
# Replace values that already exist
uv run forge train experiment=distributions/ddpm/base seed=1 runner.params.batch_size=128

# Swap a selected config group
uv run forge train experiment=distributions/ddpm/base sampler=ddim

# Add a group or key the base recipe did not define
uv run forge train experiment=distributions/ddpm/base +criterion=huber
uv run forge train experiment=distributions/ddpm/base +method.params.q=8.0
```

Config is strict by design: a bare override cannot invent a missing key. Use `+` when adding.

## Extending

Register a component, then load the module as a plugin:

```python
from forge.core.interfaces import Model
from forge.core.registry import register

@register("model", "mynet")
class MyNet(Model):
    output_type = "eps"

    def forward(self, x, t, cond=None):
        ...
```

```yaml
plugins:
  - my_package.models
model:
  name: mynet
  params: {}
```

The same pattern applies to `space`, `schedule`, `method`, `sampler`, `criterion`, `cost`,
`control`, `metric`, `runner`, `environment`, and `dataset`.

## Documentation

Serve the full documentation locally:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

The docs cover installation, architecture and invariants, experiment recipes, worked examples, and
extension patterns:

- [Installation](docs/installation.md)
- [Architecture](docs/architecture.md)
- [Experiments](docs/experiments.md)
- [Examples](docs/examples.md)
- [Extending](docs/extending.md)

## Development

```bash
uv sync --group dev
uv run pytest
uv run forge list
```

The test suite includes compile/import checks, builder and registry tests, samplers and methods,
text and robotics adapter coverage, checkpoint/resume behavior, metrics, CLI paths, and plugin
entry-point discovery.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use `genforge` in research, cite the project metadata in [CITATION.cff](CITATION.cff).

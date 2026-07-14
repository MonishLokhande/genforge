# Installation

`genforge` is managed with [`uv`](https://docs.astral.sh/uv/). There are two ways in: install
the package as a library, or clone the source for development and the bundled experiments.

---

## 1. As a library

```bash
pip install genforge          # or: uv add genforge
```

> **Note:** the PyPI wheel ships the framework only (`src/forge`). The bundled `envs/`
> data-source plugins and `experiment/` recipes stay in the source repository. To point an
> installed library at an experiment tree elsewhere on disk, set:
>
> ```bash
> export GENFORGE_EXP_ROOT="/path/to/project_root_with_experiments/"
> ```

---

## 2. From source (development & bundled experiments)

### Prerequisites

* **Python** 3.11+
* **Git** — for cloning, and for the git-pinned robotics dependency.
* **uv** — if missing, install with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone https://github.com/MonishLokhande/genforge
cd genforge
./install.sh          # installs uv if missing, then runs `uv sync`
```

Extra args pass straight through to `uv sync`, e.g. `./install.sh --extra flow --group robotics`.

This creates a `.venv` with the core framework only — enough for the 2-D distribution
experiments.

---

## 3. Optional features

The core stays light; heavier capabilities are opt-in.

### Extras

Enable with `uv sync --extra <name>`; flags can be chained (e.g. `--extra flow --extra logging`):

| Extra | Adds | For |
| --- | --- | --- |
| `flow` | `scipy` | Optimal-transport couplings (OT-CFM) and the exact Wasserstein-2 metric. |
| `text` | `tiktoken`, `datasets` | GPT-2 BPE tokenization and streamed text corpora. |
| `logging` | `wandb`, `tqdm` | Experiment logging and progress bars. |

### Robotics dependency group

The robotics environments depend on packages that don't distribute cleanly as wheel extras
(one installs from git), so they are a **dependency group** rather than an extra:

```bash
uv sync --group robotics   # mujoco, robosuite, robomimic (from git), gym-pusht/aloha, minari
```

> First sync takes a few minutes: it pulls the MuJoCo and robosuite wheels and installs
> robomimic from its git source.

### Headless rendering

The `robotics/*` environments render through MuJoCo. On servers without a display, pick the
rendering backend explicitly:

```bash
# GPU off-screen rendering (NVIDIA / AMD)
MUJOCO_GL=egl uv run forge train experiment=robotics/pusht/ddpm

# CPU software rendering (fallback)
MUJOCO_GL=osmesa uv run forge train experiment=robotics/pusht/ddpm
```

---

## 4. Verify

```bash
uv run forge list
```

If it prints the registered components by category, the environment is good.

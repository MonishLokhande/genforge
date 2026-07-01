"""Optional experiment logging — a tiny Logger protocol with a null default and a lazily
imported wandb-backed impl, plus a progress-bar wrapper.

``wandb`` and ``tqdm`` are an OPTIONAL extra (``pip install genforge[logging]``); without it you
get a no-op logger and a plain loop, never an ImportError. wandb is imported ONLY inside
``WandbLogger.__init__`` so it never enters the core import graph. Runners call ``logger.log(...)``
UNCONDITIONALLY — the null default absorbs it — so there are no ``if logger:`` guards in the loop.

``make_logger`` is loop-agnostic (it takes only ``log``/``run_name``/``config``), so the same
factory serves the training loop today and an eval/verification path later. The ``Logger`` surface
is deliberately minimal (``log`` + ``finish``); richer impls may add ``log_*`` methods (e.g.
``log_scatter``, ``log_figure`` for the 2-D distribution eval path) and ``NullLogger`` no-ops any
such call, so the extension is additive — no call site changes when it lands.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Minimal experiment-logging surface. Impls may add optional ``log_*`` methods later."""

    def log(self, metrics: dict, step: Optional[int] = None) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    """Default no-op logger. Every call — including future ``log_*`` methods — is a no-op, so
    runners call ``logger.log(...)`` unconditionally and richer impls stay additive."""

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        pass

    def finish(self) -> None:
        pass

    def __getattr__(self, name: str):
        # Absorb any future richer call (log_scatter / log_figure / log_artifacts / ...) as a
        # no-op, but still raise for genuine typos on non-logging attributes.
        if name.startswith("log"):
            return lambda *a, **k: None
        raise AttributeError(name)


class WandbLogger:
    """Thin Weights & Biases wrapper. ``wandb`` is imported lazily here — never at module load.

    NOTE: wandb enforces a monotonically increasing ``step`` within a run. The training loop's
    step is monotonic, so v1 is safe. A future eval/verification path that logs to the *same* run
    should use a separate run or a step offset rather than assume one monotonic caller.
    """

    def __init__(self, *, project: str, name: Optional[str] = None,
                 config: Optional[dict] = None, mode: Optional[str] = None) -> None:
        import wandb  # lazy: only when a WandbLogger is actually constructed
        self.wandb = wandb
        # reinit="finish_previous": close any open run first so repeated in-process runs
        # (e.g. a notebook re-execution) don't error.
        self.run = wandb.init(project=project, name=name, config=config, mode=mode,
                              reinit="finish_previous")

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()


def cfg_get(log: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an OmegaConf node / plain dict / None ``log`` config, safely."""
    if log is None:
        return default
    if hasattr(log, "get"):
        return log.get(key, default)
    return getattr(log, key, default)


def make_logger(log: Any = None, *, run_name: Optional[str] = None,
                config: Optional[dict] = None) -> Logger:
    """Build a Logger from a runner's ``log`` config. Loop-agnostic — usable from train or eval.

    wandb is OFF unless ``log.wandb`` is true (or env ``FORGE_WANDB=1``). ANY failure — the extra
    not installed, or wandb not logged in — degrades to ``NullLogger`` with a printed note, never a
    crash and never an interactive login prompt on a default run.
    """
    enabled = bool(cfg_get(log, "wandb", False)) or os.environ.get("FORGE_WANDB") == "1"
    if not enabled:
        return NullLogger()
    project = cfg_get(log, "project") or os.environ.get("WANDB_PROJECT") or "forge"
    mode = cfg_get(log, "mode")            # online | offline | disabled (None → wandb default)
    name = cfg_get(log, "name") or run_name
    try:
        return WandbLogger(project=project, name=name, config=config, mode=mode)
    except Exception as exc:  # not installed / not logged in / init failure → degrade
        print(f"[log] wandb disabled ({type(exc).__name__}: {exc})")
        return NullLogger()


def progress_iter(iterable: Iterable, *, enable: bool = False, **tqdm_kwargs) -> Iterator:
    """Wrap ``iterable`` in a tqdm bar when enabled AND safe; else return a plain iterator.

    Degrades to a plain loop when: ``enable`` is false, ``stderr`` is not a TTY (piped / redirected
    / CI / notebook-without-widget — a bar there spams carriage returns into logs), or ``tqdm`` is
    not installed. One wrapper — no bare ``tqdm()`` calls in loops.
    """
    if not enable or not sys.stderr.isatty():
        return iter(iterable)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iter(iterable)
    return tqdm(iterable, **tqdm_kwargs)


__all__ = ["Logger", "NullLogger", "WandbLogger", "make_logger", "progress_iter", "cfg_get"]

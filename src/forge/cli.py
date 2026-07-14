"""The ``forge`` command-line entrypoint: ``list`` / ``train`` / ``sample`` / ``eval``.

``list`` imports the built-ins so registrations fire, then prints the registered components by
category. ``train`` / ``sample`` compose a Hydra config from an ``experiment=`` selection, build the
runner, and run. ``sample checkpoint=<path.pt>`` rebuilds everything from the self-contained
checkpoint alone (Invariant 5). All three fail loudly on misconfiguration.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .core import registry
from .core.builder import build, import_builtin_components

_EXPERIMENT_HINT = (
    "Select one with `experiment=<env>/<params>/<method>` "
    "(e.g. `forge train experiment=distributions/ddpm/base`)."
)


def _split_overrides(overrides: Sequence[str]) -> dict:
    """Parse ``key=value`` overrides into a flat dict (first '=')."""
    flat: dict[str, str] = {}
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            flat[k] = v
    return flat


def _cmd_list(_args: argparse.Namespace) -> int:
    import_builtin_components()
    # Concrete envs are plugins (no experiment selected here), so import the bundled env packages
    # too — otherwise `list` would omit environments/datasets/env-preprocessors.
    from .core.plugins import load_bundled_envs

    load_bundled_envs()
    reg = registry.registered()
    print("forge components")
    print("===================")
    for category in registry.CATEGORIES:
        comps = reg.get(category, {})
        names = ", ".join(comps) if comps else "(none yet)"
        print(f"  {category:<13} {names}")
    for category in [c for c in reg if c not in registry.CATEGORIES]:
        print(f"  {category:<13} {', '.join(reg[category])}")
    return 0


def _run_from_config(overrides: Sequence[str], action: str) -> int:
    from omegaconf import OmegaConf

    from .core.compose import compose_config

    cfg = compose_config(overrides)
    runner = build(cfg)
    runner.resolved_config = OmegaConf.to_container(cfg, resolve=True)

    if action == "train":
        runner.train()
        metrics = runner.evaluate()
        print(f"[train] done. eval: {metrics}")
        return 0

    # sample/eval from an experiment: load its configured checkpoint if present.
    ckpt_path = getattr(runner, "ckpt_path", None)
    if ckpt_path:
        from pathlib import Path

        from .core.checkpoint import load_checkpoint

        if not Path(ckpt_path).exists():
            print(
                f"`forge {action}` found no checkpoint at {ckpt_path!r}. Train first "
                f"(`forge train experiment=...`) or pass `checkpoint=<path.pt>`.",
                file=sys.stderr,
            )
            return 1
        runner.load_state(load_checkpoint(ckpt_path))
    metrics = runner.evaluate()
    print(f"[{action}] {metrics}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    flat = _split_overrides(args.overrides)
    if "experiment" not in flat:
        print(f"`forge train` requires an experiment selection. {_EXPERIMENT_HINT}", file=sys.stderr)
        return 2
    return _run_from_config(args.overrides, "train")


def _cmd_sample(args: argparse.Namespace) -> int:
    flat = _split_overrides(args.overrides)
    if "checkpoint" in flat:
        # Self-contained path: rebuild from the .pt alone (Invariant 5).
        from .runners.training import TrainingRunner

        runner = TrainingRunner.from_checkpoint(flat["checkpoint"], build_fn=build)
        metrics = runner.evaluate()
        print(f"[sample] from checkpoint {flat['checkpoint']}: {metrics}")
        return 0
    if "experiment" not in flat:
        print(
            f"`forge sample` requires `experiment=...` or `checkpoint=<path.pt>`. {_EXPERIMENT_HINT}",
            file=sys.stderr,
        )
        return 2
    return _run_from_config(args.overrides, "sample")


def _cmd_eval(args: argparse.Namespace) -> int:
    flat = _split_overrides(args.overrides)
    if "samples" in flat:
        # Offline: score a saved samples file — build a LIGHTWEIGHT pipeline (no weight load, no
        # sampling), run sample-driven metrics against the env reference. Data-driven metrics
        # (which need the model + held-out data) RAISE — direct the user to `checkpoint=`.
        from omegaconf import OmegaConf

        from .core.checkpoint import load_checkpoint
        from .utils.persistence import load_samples, save_metrics

        if "experiment" in flat:
            from .core.compose import compose_config

            # Strip CLI-only keys (samples=/checkpoint=) — they are not Hydra config overrides.
            hydra = [o for o in args.overrides if o.split("=", 1)[0] not in ("samples", "checkpoint")]
            cfg = compose_config(hydra)
            runner = build(cfg)
            runner.resolved_config = OmegaConf.to_container(cfg, resolve=True)
        elif "checkpoint" in flat:
            runner = build(load_checkpoint(flat["checkpoint"])["config"])  # config only, no weights
        else:
            print("`forge eval samples=<file>` also needs `experiment=...` or `checkpoint=<.pt>` "
                  "to build the env + metric.", file=sys.stderr)
            return 2
        if runner.metric is None:
            print("`forge eval samples=` needs a metric configured (a `metric` leaf in the "
                  "experiment/checkpoint).", file=sys.stderr)
            return 2
        samples = load_samples(flat["samples"])
        metrics = {"n": float(samples.shape[0])}
        metrics.update(runner.metric(samples=samples, held_out=None))  # data-driven → raises (fail-loud)
        out_dir = runner._output_dir()
        if out_dir is not None:
            save_metrics(out_dir, metrics, step=getattr(runner, "_completed_steps", 0))
        print(f"[eval] samples={flat['samples']}: {metrics}")
        return 0
    if "checkpoint" in flat:
        from .runners.training import TrainingRunner

        runner = TrainingRunner.from_checkpoint(flat["checkpoint"], build_fn=build)
        print(f"[eval] from checkpoint {flat['checkpoint']}: {runner.evaluate()}")
        return 0
    if "experiment" not in flat:
        print(f"`forge eval` requires `experiment=...`, `checkpoint=<.pt>`, or `samples=<file>` "
              f"(with experiment=/checkpoint=). {_EXPERIMENT_HINT}", file=sys.stderr)
        return 2
    return _run_from_config(args.overrides, "eval")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="A unified framework for generative modeling with a clean control layer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List registered components by category.")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="Train a model from an experiment config.")
    p_train.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. experiment=...")
    p_train.set_defaults(func=_cmd_train)

    p_sample = sub.add_parser("sample", help="Sample from a trained model or checkpoint.")
    p_sample.add_argument("overrides", nargs="*", help="experiment=... or checkpoint=<path.pt>")
    p_sample.set_defaults(func=_cmd_sample)

    p_eval = sub.add_parser("eval", help="Score a checkpoint/experiment, or a saved samples file.")
    p_eval.add_argument("overrides", nargs="*",
                        help="experiment=... | checkpoint=<.pt> | samples=<file> (+experiment/checkpoint)")
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

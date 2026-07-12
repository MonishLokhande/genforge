"""The training runner: train in normalized space, sample with EMA, write self-contained ckpts.

Lifecycle:
  - fit the preprocessor once on the full training set (if present), then train entirely inside the
    membrane (normalized coordinates);
  - sample with the EMA shadow, then invert the preprocessor on the output;
  - save a self-contained checkpoint (weights + EMA + preprocessor stats + resolved config +
    provenance + optimizer/scheduler/RNG state) that can `sample` on its own and **resume
    bit-identically** on the preloaded fast path.

All randomness flows through two explicit generators (batch indices, training noise), so a
checkpoint captures the complete RNG state and `train(resume_from=...)` continues the exact same
trajectory. The runner never touches α/σ or output-type math and never branches on cont/disc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from ..core.checkpoint import (
    build_checkpoint,
    current_git_hash,
    load_checkpoint,
    save_checkpoint,
)
from ..core.interfaces import Runner
from ..core.protocols import BaseDataset
from ..core.registry import register
from ..core.types import Provenance
from ..utils.ema import EMA
from ..utils.logging import cfg_get, make_logger, progress_iter
from ..utils.lora import LoRALinear, apply_lora
from ..utils.seeding import make_generator, set_seed


@register("runner", "training")
class TrainingRunner(Runner):
    def __init__(
        self,
        model,
        method,
        sampler,
        space,
        schedule,
        dataset: BaseDataset,
        environment=None,
        preprocessor=None,
        visualizer=None,
        *,
        steps: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Optional[str] = None,   # None | "cosine"
        lr_min_ratio: float = 0.05,
        ema_decay: float = 0.999,
        ema_warmup: int = 10,
        grad_clip: Optional[float] = None,
        n_sample_steps: int = 100,
        n_eval_samples: int = 2000,
        eval_radius: float = 0.6,
        device: str = "cpu",
        seed: int = 0,
        sample_seed: int = 12345,
        deterministic: bool = False,
        amp: bool = False,                  # bf16 autocast on loss + sampling; bf16 → no GradScaler
        log_every: int = 200,
        ckpt_path: Optional[str] = None,
        warm_start: Optional[str] = None,   # weights-only load from a .pt, fresh optimizer (≠ resume)
        lora: Optional[Any] = None,         # {r, alpha, dropout, targets} → freeze base, train adapters
        log: Optional[Any] = None,          # {wandb, project, mode, name, progress} — optional logging extra
    ):
        self.model = model
        self.method = method
        self.sampler = sampler
        self.space = space
        self.schedule = schedule
        self.dataset = dataset
        self.environment = environment
        self.preprocessor = preprocessor
        self.visualizer = visualizer

        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.ema_decay = ema_decay
        self.ema_warmup = ema_warmup
        self.grad_clip = grad_clip
        self.n_sample_steps = n_sample_steps
        self.n_eval_samples = n_eval_samples
        self.eval_radius = eval_radius
        self.device = device
        self.seed = seed
        self.sample_seed = sample_seed
        self.deterministic = deterministic
        self.amp = amp
        self.log_every = log_every
        self.ckpt_path = ckpt_path
        self.warm_start = warm_start
        self.lora = lora
        self.log = log

        self.dim = getattr(dataset, "dim", None) or getattr(space, "dim", None)
        self.ema: Optional[EMA] = None
        self.resolved_config: Any = None
        self._lora_config: Optional[dict] = None   # resolved {r,alpha,...}, travels in the checkpoint
        self._last_losses: list[float] = []
        # Live training state (set during train(); persisted for resume).
        self._opt: Optional[torch.optim.Optimizer] = None
        self._lr_sched = None
        self._batch_gen: Optional[torch.Generator] = None
        self._noise_gen: Optional[torch.Generator] = None
        self._completed_steps: int = 0

    # ── training ────────────────────────────────────────────────────────────────────────────────
    def _fit_model_conditioning(self, device: torch.device) -> None:
        """Fit any in-model conditioning normalizer (Inv 9: obs normalization is the model's job,
        not the membrane). Duck-typed — fires only when the model exposes an ``obs_normalizer`` AND
        the dataset a ``cond_fit_tensor``; a no-op for unconditional / janner / value setups."""
        obs_norm = getattr(self.model, "obs_normalizer", None)
        cond_fit = getattr(self.dataset, "cond_fit_tensor", None)
        if obs_norm is not None and cond_fit is not None:
            obs_norm.fit(cond_fit.to(device))

    def _setup_lora(self, cfg: Optional[Any] = None) -> None:
        """Wrap the model's target Linears with LoRA adapters and freeze the base (Inv-free —
        a model-architecture step). Idempotent. Prefers ``self.lora`` (live config), else ``cfg``
        (a checkpoint's stored ``lora_config``) so adapters reconstruct at load time BEFORE
        ``load_state_dict``. No-op when neither is set."""
        spec = self.lora if self.lora is not None else cfg
        if not spec:
            return
        if any(isinstance(m, LoRALinear) for m in self.model.modules()):
            return  # already wrapped (e.g. load_state after a prior setup) — don't double-apply
        from omegaconf import OmegaConf  # local: avoid a hard import when LoRA is unused

        spec = spec if isinstance(spec, dict) else OmegaConf.to_container(spec, resolve=True)
        self._lora_config = dict(spec)
        n = apply_lora(self.model, **spec)
        print(f"[train] LoRA: wrapped {n} layer(s) (r={spec.get('r', 8)}, targets matched).")

    def train(self, resume_from: Optional[str | dict] = None) -> None:
        set_seed(self.seed, self.deterministic)
        device = torch.device(self.device)
        self.model.to(device)

        # Fit the membrane once on the per-feature fit tensor (flat trajectory tensor for windows).
        if self.preprocessor is not None:
            self.preprocessor.fit(self.dataset.fit_tensor.to(device))
        self._fit_model_conditioning(device)
        n = self.dataset.num_items

        # Warm-start (≠ resume): load *vanilla base* weights only, then a fresh optimizer trains
        # from step 1 on the new data. Must run BEFORE _setup_lora so the base keys match (a LoRA
        # model's keys are `…base.weight`); resume_from (below) is a LoRA-aware same-trajectory load.
        if self.warm_start is not None:
            if resume_from is not None:
                raise ValueError("Pass either warm_start or resume_from, not both (distinct intents).")
            ws = load_checkpoint(self.warm_start)
            inc = self.model.load_state_dict(ws["model_state"], strict=False)
            if inc.missing_keys or inc.unexpected_keys:
                print(
                    f"[train] warm_start {self.warm_start}: load_state_dict strict=False — "
                    f"missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)} "
                    f"(e.g. a vocab/head change reinitializes those)."
                )
        self._setup_lora()
        n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        if self.lora and n_frozen > 5_000_000:
            print(
                f"[train] EMA tracks {n_frozen:,} frozen params — consider ema_decay=0 / "
                f"disabling EMA for large LoRA bases."
            )

        self.ema = EMA(self.model, self.ema_decay, warmup=self.ema_warmup)
        # Optimizer sees only trainable params — a no-op when nothing is frozen, required for LoRA.
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay,
        )
        lr_sched = None
        if self.lr_schedule == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.steps, eta_min=self.lr * self.lr_min_ratio
            )
        elif self.lr_schedule not in (None, "none"):
            raise ValueError(f"Unknown lr_schedule {self.lr_schedule!r} (expected None or 'cosine').")

        # All training randomness flows through these two generators (Invariant 5 resumability).
        batch_gen = make_generator(self.seed + 1, device)
        noise_gen = make_generator(self.seed + 2, device)

        start_step = 1
        if resume_from is not None:
            ckpt = resume_from if isinstance(resume_from, dict) else load_checkpoint(resume_from)
            self.model.load_state_dict(ckpt["model_state"])
            if ckpt.get("ema_state") is not None:
                self.ema.load_state_dict(ckpt["ema_state"])
                self._relocate_ema(device)
            if ckpt.get("optimizer_state") is not None:
                opt.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None and lr_sched is not None:
                lr_sched.load_state_dict(ckpt["scheduler_state"])
            rng = ckpt.get("rng_state") or {}
            if "batch_gen" in rng:
                batch_gen.set_state(rng["batch_gen"].to("cpu"))
            if "noise_gen" in rng:
                noise_gen.set_state(rng["noise_gen"].to("cpu"))
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].to("cpu"))
            start_step = int(ckpt.get("step", 0)) + 1

        self._opt, self._lr_sched = opt, lr_sched
        self._batch_gen, self._noise_gen = batch_gen, noise_gen

        self.model.train()
        self._last_losses = []
        # Optional experiment logging (wandb) + progress bar — both no-op without the `logging`
        # extra, so a default run is byte-identical to before. make_logger is loop-agnostic.
        wb_config = None
        if self.resolved_config is not None:
            try:  # never let config serialization for the logger break training
                from omegaconf import OmegaConf
                wb_config = (
                    OmegaConf.to_container(self.resolved_config, resolve=True)
                    if OmegaConf.is_config(self.resolved_config)
                    else (dict(self.resolved_config) if isinstance(self.resolved_config, dict) else None)
                )
            except Exception:
                wb_config = None
        run_name = f"{getattr(self.method, 'name', 'run')}-{self.steps}steps"
        logger = make_logger(self.log, run_name=run_name, config=wb_config)
        step_iter = progress_iter(
            range(start_step, self.steps + 1),
            enable=bool(cfg_get(self.log, "progress", False)),
            desc="train", total=self.steps, initial=start_step - 1,
        )
        for step in step_iter:
            idx = torch.randint(0, n, (self.batch_size,), generator=batch_gen, device=device)
            batch = self.dataset.batch(idx)            # BatchProtocol — the batch's first entry point
            BaseDataset.validate_batch(batch)          # enforce the contract BEFORE the membrane
            x0 = batch.x0.to(device)
            if self.preprocessor is not None:
                x0 = self.preprocessor.transform(x0)   # membrane touches the generated quantity only
            # Conditioning rides to the model's device but is NEVER pushed through the membrane (Inv 9).
            cond = batch.cond
            if torch.is_tensor(cond):
                cond = cond.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.amp):
                loss = self.method.loss(self.model, x0, cond=cond, generator=noise_gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()
            if lr_sched is not None:
                lr_sched.step()
            self.ema.update(self.model)

            self._last_losses.append(loss.detach())   # defer the device→host sync out of the hot loop
            self._completed_steps = step
            if self.log_every and (step % self.log_every == 0 or step == start_step):
                lv = loss.item()   # one device→host sync on the log cadence (reused for both sinks)
                print(f"[train] step {step}/{self.steps}  loss={lv:.4f}")
                logger.log({"loss": lv, "lr": opt.param_groups[0]["lr"]}, step=step)
        logger.finish()

        # Materialize per-step losses to python floats once, after the loop (one bulk sync, not 1/step).
        self._last_losses = [float(x) for x in self._last_losses]

        if self.ckpt_path:
            self.save_checkpoint(self.ckpt_path)

    # ── sampling ────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: Optional[int] = None, cond=None) -> torch.Tensor:
        device = torch.device(self.device)
        self.model.to(device)
        self.model.eval()

        used_ema = self.ema is not None
        if used_ema:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        # Push a real-unit cost into normalized space now that the membrane is fitted/loaded (Inv 8).
        control = getattr(self.sampler, "control", None)
        if control is not None and hasattr(control, "prepare"):
            control.prepare(self.preprocessor)

        n = n or self.n_eval_samples
        gen = make_generator(self.sample_seed, device)
        shape = (n, *self.dataset.sample_shape)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.sampler.sample(shape, self.n_sample_steps, cond=cond, generator=gen)
        x = out.samples
        if self.preprocessor is not None:
            x = self.preprocessor.inverse(x)

        if used_ema:
            self.ema.restore(self.model)
        return x

    def evaluate(self) -> dict:
        radius = self.eval_radius
        x = self.sample()
        env = self.environment
        # Metric logic is delegated to the data source when it provides one (capability check, not
        # a cont/disc branch) — this keeps the runner agnostic across paradigms.
        if env is not None and hasattr(env, "evaluate"):
            return env.evaluate(x)
        metrics: dict[str, float] = {"n": float(x.shape[0])}
        if env is not None and hasattr(env, "means"):
            means = env.means.to(x.device)
            nearest = torch.cdist(x, means).min(dim=1).values
            metrics["mode_coverage"] = float((nearest < radius).float().mean().item())
            metrics["radius"] = float(radius)
        return metrics

    # ── checkpoint ──────────────────────────────────────────────────────────────────────────────
    def _relocate_ema(self, device: torch.device) -> None:
        if self.ema is not None:
            self.ema.shadow = {k: v.to(device) for k, v in self.ema.shadow.items()}

    def _rng_state(self) -> Optional[dict]:
        if self._batch_gen is None or self._noise_gen is None:
            return None
        return {
            "batch_gen": self._batch_gen.get_state().cpu(),
            "noise_gen": self._noise_gen.get_state().cpu(),
            "torch": torch.get_rng_state(),
        }

    def save_checkpoint(self, path: Optional[str] = None) -> dict:
        ckpt = build_checkpoint(
            model_state=self.model.state_dict(),
            ema_state=self.ema.state_dict() if self.ema is not None else None,
            preprocessor_state=(
                self.preprocessor.state_dict() if self.preprocessor is not None else None
            ),
            config=self.resolved_config,
            provenance=Provenance(git_hash=current_git_hash(), seed=self.seed),
            optimizer_state=self._opt.state_dict() if self._opt is not None else None,
            scheduler_state=self._lr_sched.state_dict() if self._lr_sched is not None else None,
            rng_state=self._rng_state(),
            lora_config=self._lora_config,
            step=self._completed_steps,
        )
        path = path or self.ckpt_path
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(path, ckpt)
        return ckpt

    def load_state(self, ckpt: dict) -> None:
        """Restore weights + EMA + preprocessor stats from a self-contained checkpoint."""
        device = torch.device(self.device)
        self.model.to(device)
        # Reconstruct LoRA adapters BEFORE loading weights so the `…base.weight`/`.A`/`.B` keys
        # exist (uses the checkpoint's stored lora_config when the live config lacks one).
        self._setup_lora(ckpt.get("lora_config"))
        if ckpt.get("model_state") is not None:
            self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("ema_state") is not None:
            self.ema = EMA(self.model, self.ema_decay, warmup=self.ema_warmup)
            self.ema.load_state_dict(ckpt["ema_state"])
            self._relocate_ema(device)
        if ckpt.get("preprocessor_state") is not None and self.preprocessor is not None:
            self.preprocessor.load_state_dict(ckpt["preprocessor_state"])

    @classmethod
    def from_checkpoint(cls, path: str, build_fn) -> "TrainingRunner":
        """Build a runner from a checkpoint's embedded config and restore its state.

        ``build_fn(config) -> runner`` is the builder (passed in to avoid an import cycle).
        """
        ckpt = load_checkpoint(path)
        config = ckpt.get("config")
        if config is None:
            raise ValueError(f"Checkpoint {path} has no embedded config; cannot rebuild.")
        runner = build_fn(config)
        runner.load_state(ckpt)
        return runner

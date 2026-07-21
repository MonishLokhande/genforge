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
from ..utils.torch_utils import cond_to


class _StepIndexSampler:
    """One ``(batch_size,)`` index tensor per training step, derived from ``(seed, step)``.

    PURE in ``step`` — a fresh generator per step, so the stream never depends on generator
    POSITION. That is the whole trick: a DataLoader's main process fills the worker prefetch queue
    AHEAD of the loop, so wrapping the runner's stateful ``batch_gen`` in a sampler would leave it
    advanced by ``num_workers * prefetch_factor``; a checkpoint at step N would then store the state
    for step N+drift and resume would silently SKIP those batches. Deriving from ``step`` makes the
    drift irrelevant and leaves nothing to checkpoint — ``step`` is already in the checkpoint
    (Invariant 5).

    REQUIRES ``dataset.batch(idx)`` to be PURE in ``idx``: any randomness inside it would come from
    worker RNG, which is not a function of ``step`` and is not checkpointable. Augmentation belongs
    in the runner's main-process ``dataset.augment`` hook instead.
    """

    def __init__(self, n: int, pool, batch_size: int, seed: int, start_step: int, steps: int,
                 grad_accum: int = 1):
        self.n, self.pool, self.batch_size = n, pool, int(batch_size)
        self.seed, self.start_step, self.steps = int(seed), int(start_step), int(steps)
        self.grad_accum = int(grad_accum)     # gradient accumulation yields grad_accum micro-batches/step

    def __iter__(self):
        hi = self.n if self.pool is None else self.pool.numel()
        # Iterate in MICRO-steps: one index draw per micro-batch, still pure in the micro index. At
        # grad_accum=1 this is `range(start_step, steps+1)` with seed `…+step` — byte-identical.
        start_micro = (self.start_step - 1) * self.grad_accum + 1
        total_micro = self.steps * self.grad_accum
        for micro in range(start_micro, total_micro + 1):
            g = torch.Generator()                                  # CPU: index draws are host-side
            g.manual_seed((self.seed * 1_000_003 + micro) & (2**63 - 1))   # arithmetic, not hash()
            sel = torch.randint(0, hi, (self.batch_size,), generator=g)
            yield sel if self.pool is None else self.pool[sel]     # mirrors the fast path's val pool

    def __len__(self) -> int:
        return max(0, (self.steps - self.start_step + 1) * self.grad_accum)


class _BatchView:
    """Adapts a genforge dataset to `DataLoader` so ONE 'item' IS a whole batch.

    With ``batch_size=None`` the loader disables automatic batching and passes each sampler item
    straight to ``__getitem__`` — so the worker runs the dataset's own batched fetch. This reuses
    ``batch(idx)`` as the fetch unit: no per-sample ``__getitem__``, no ``collate_fn``. It is also
    the only unit an h5py-backed dataset can sort its fancy index in (h5py requires strictly
    increasing indices).
    """

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds.batch(idx)

    def __len__(self) -> int:
        return self.ds.num_items
from ..utils.persistence import save_metrics, save_samples
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
        metric=None,
        *,
        steps: int = 2000,
        batch_size: int = 256,
        grad_accum: int = 1,                # micro-batches summed per optimizer step; effective batch
                                            # = batch_size * grad_accum. 1 = disabled (byte-identical)
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),         # Adam/AdamW moments; transformer training wants beta2≈0.95
        weight_decay: float = 0.0,
        optimizer: str = "adamw",            # "adamw" (decoupled) | "adam" (coupled L2); else → own runner
        decay_1d: bool = True,               # decay 1-D params (biases, LayerNorm/RMSNorm gains) too;
                                             # False → nanoGPT-style exclusion (only bites when wd>0)
        lr_schedule: Optional[str] = None,   # None | "cosine"
        lr_min_ratio: float = 0.05,
        warmup_steps: int = 0,               # linear LR warmup before the schedule (0 = off)
        ema_decay: float = 0.999,
        ema_warmup: int = 10,
        grad_clip: Optional[float] = None,
        n_sample_steps: int = 100,
        n_eval_samples: int = 2000,
        eval_radius: float = 0.6,
        val_frac: float = 0.0,              # held-out fraction; 0 = no split (train uses all items)
        val_every: int = 0,                 # steps between held-out val-loss passes (0 = off)
        eval_every: int = 0,                # steps between full evaluate() (sample+metrics) (0 = off)
        save_best: bool = False,            # keep <ckpt>.best.pt at the lowest val_loss
        device: str = "cpu",
        seed: int = 0,
        sample_seed: int = 12345,
        deterministic: bool = False,
        amp: bool = False,                  # bf16 autocast on loss + sampling; bf16 → no GradScaler
        compile: bool = False,              # torch.compile the training-forward path (raw model kept for EMA/ckpt)
        workers: int = 0,                   # DataLoader workers; ONLY used when the dataset opts
                                            # out of the fast path (supports_fast_path=False)
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
        self.metric = metric

        self.steps = steps
        self.batch_size = batch_size
        self.grad_accum = int(grad_accum)
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1, got {grad_accum}.")
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.decay_1d = decay_1d
        self.lr_schedule = lr_schedule
        self.lr_min_ratio = lr_min_ratio
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.ema_warmup = ema_warmup
        self.grad_clip = grad_clip
        self.n_sample_steps = n_sample_steps
        self.n_eval_samples = n_eval_samples
        self.eval_radius = eval_radius
        self.val_frac = float(val_frac)
        self.val_every = int(val_every)
        self.eval_every = int(eval_every)
        self.save_best = bool(save_best)
        self._train_idx = None              # set only when val_frac>0 (else draw from all items)
        self._val_idx = None
        self.device = device
        self.seed = seed
        self.sample_seed = sample_seed
        self.deterministic = deterministic
        self.amp = amp
        self.compile = compile
        self.workers = workers
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
        self._aug_gen: Optional[torch.Generator] = None
        self._completed_steps: int = 0

    # ── training ────────────────────────────────────────────────────────────────────────────────
    def _make_batch_source(self, n: int, batch_gen, device, start_step: int):
        """A uniform ``next_batch()`` closure, choosing on the dataset's CAPABILITY — never on what
        the data *is*. So an out-of-RAM lowdim dataset gets streaming for free, and no image-vs-lowdim
        branch ever enters the runner.

        ``supports_fast_path`` defaults True via ``getattr``, so every existing dataset keeps the
        in-RAM index path verbatim (it is ~4.5x faster than a DataLoader on small data — the reason
        genforge has no loader at all). A dataset that cannot be preloaded says so, and the runner
        decides what to do about it; the dataset never constructs a loader itself.
        """
        if getattr(self.dataset, "supports_fast_path", True):
            def next_batch():
                if self._train_idx is None:            # val_frac=0: draw from all items (byte-identical)
                    idx = torch.randint(0, n, (self.batch_size,), generator=batch_gen, device=device)
                else:                                  # val_frac>0: draw from the train pool only
                    sel = torch.randint(0, self._train_idx.numel(), (self.batch_size,),
                                        generator=batch_gen, device=device)
                    idx = self._train_idx[sel]
                return self.dataset.batch(idx)
            return next_batch

        from torch.utils.data import DataLoader

        workers = int(self.workers or 0)
        loader = DataLoader(
            _BatchView(self.dataset),
            sampler=_StepIndexSampler(n, self._train_idx, self.batch_size, self.seed + 1,
                                      start_step, self.steps, self.grad_accum),
            batch_size=None,                    # one sampler item == one batch (see _BatchView)
            collate_fn=lambda b: b,             # already assembled; do not touch it
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=(2 if workers > 0 else None),
        )
        # No worker_init_fn/seed_worker: the index stream is pure in (seed, step) and the fetch is
        # pure in idx, so no worker owns randomness and there is nothing to seed or checkpoint.
        stream = iter(loader)
        return lambda: next(stream)

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

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        """Assemble the optimizer from the runner's knobs (the design contract).

        - `optimizer`: 'adamw' (decoupled decay — the default) | 'adam' (weight_decay folded into the
          gradient as coupled L2). Anything else fails loudly: register your own runner (Invariant 7).
        - `betas`: passed through — transformer training usually wants beta2≈0.95, not Adam's 0.999.
        - `weight_decay`: when `decay_1d` is False it is EXCLUDED from 1-D params (biases,
          LayerNorm/RMSNorm gains) via two param groups — the nanoGPT convention. No-op when wd==0.
        """
        try:
            opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[self.optimizer.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}; choose from 'adam', 'adamw'. "
                f"For any other optimizer, register a new runner (Invariant 7)."
            )
        if self.optimizer.lower() == "adam" and self.weight_decay > 0:
            print(
                f"[train] optimizer=adam folds weight_decay={self.weight_decay} into the gradient as "
                f"coupled L2, which interacts poorly with Adam's per-parameter scaling. Prefer "
                f"optimizer=adamw for decoupled decay (AdamW, 2019)."
            )
        params = list(params)
        betas = tuple(self.betas)
        if self.weight_decay > 0 and not self.decay_1d:
            decay = [p for p in params if p.dim() >= 2]
            no_decay = [p for p in params if p.dim() < 2]
            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": self.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return opt_cls(groups, lr=self.lr, betas=betas)
        return opt_cls(params, lr=self.lr, betas=betas, weight_decay=self.weight_decay)

    def _build_scheduler(self, opt):
        """Optional cosine schedule, optionally prefixed by a linear LR warmup. None => flat LR."""
        S = torch.optim.lr_scheduler
        base = None
        if self.lr_schedule == "cosine":
            base = S.CosineAnnealingLR(
                opt, T_max=max(1, self.steps - self.warmup_steps),
                eta_min=self.lr * self.lr_min_ratio,
            )
        elif self.lr_schedule not in (None, "none"):
            raise ValueError(f"Unknown lr_schedule {self.lr_schedule!r} (expected None or 'cosine').")
        if self.warmup_steps > 0:
            warm = S.LinearLR(opt, start_factor=1.0 / self.warmup_steps, total_iters=self.warmup_steps)
            if base is None:
                return warm                                    # warmup, then hold at full lr
            return S.SequentialLR(opt, [warm, base], milestones=[self.warmup_steps])
        return base

    def train(self, resume_from: Optional[str | dict] = None) -> None:
        set_seed(self.seed, self.deterministic)
        device = torch.device(self.device)
        self.model.to(device)

        # Fit the membrane once on the per-feature fit tensor (flat trajectory tensor for windows).
        if self.preprocessor is not None:
            self.preprocessor.fit(self.dataset.fit_tensor.to(device))
        self._fit_model_conditioning(device)
        n = self.dataset.num_items
        self._ensure_split(n, device)   # builds train/val index pools ONLY when val_frac>0 (Inv 5)

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
        opt = self._build_optimizer([p for p in self.model.parameters() if p.requires_grad])
        lr_sched = self._build_scheduler(opt)

        # All training randomness flows through these generators (Invariant 5 resumability).
        batch_gen = make_generator(self.seed + 1, device)
        noise_gen = make_generator(self.seed + 2, device)
        # Augmentation runs in the MAIN process off its own dedicated stream. Keeping it here (not
        # inside dataset.batch()) is what makes it resumable: the loop is the only consumer, so
        # generator position always matches consumption. A stochastic batch() would instead draw
        # from worker RNG — not a function of `step`, not checkpointable — and silently desync
        # resume (Invariant 5).
        aug_gen = make_generator(self.seed + 5, device)
        # Held-out streams are DEDICATED and built only when a split exists, so val_frac=0 touches
        # no generator (the val pass must never consume batch_gen/noise_gen — Invariant 5).
        val_batch_gen = make_generator(self.seed + 3, device) if self._val_idx is not None else None
        val_noise_gen = make_generator(self.seed + 4, device) if self._val_idx is not None else None
        best_val = float("inf")

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
            if "aug_gen" in rng:        # absent in checkpoints written before augmentation existed
                aug_gen.set_state(rng["aug_gen"].to("cpu"))
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].to("cpu"))
            start_step = int(ckpt.get("step", 0)) + 1

        self._opt, self._lr_sched = opt, lr_sched
        self._batch_gen, self._noise_gen, self._aug_gen = batch_gen, noise_gen, aug_gen
        # Duck-typed like cond_fit_tensor: a dataset may augment its cond (e.g. random-crop on
        # camera frames). None for every dataset that doesn't define it — zero cost, no branching.
        augment = getattr(self.dataset, "augment", None)

        self.model.train()
        # torch.compile wraps the SAME parameters; keep self.model raw so EMA / optimizer / checkpoint
        # never see an `_orig_mod.` prefix (which would break sample-from-.pt). Only the training-forward
        # path uses the compiled callable; sampling stays on the raw (EMA-swapped) model.
        train_model = torch.compile(self.model) if self.compile else self.model
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
        next_batch = self._make_batch_source(n, batch_gen, device, start_step)
        for step in step_iter:
            opt.zero_grad(set_to_none=True)
            loss = None                                # accumulated effective-batch loss (for logging)
            # Sum grads over grad_accum micro-batches, then ONE optimizer step. grad_accum=1 is the
            # single-batch loop verbatim (micro/1 is exact; zero_grad→backward→step order unchanged).
            for _ in range(self.grad_accum):
                batch = next_batch()                   # BatchProtocol — the batch's first entry point
                BaseDataset.validate_batch(batch)      # enforce the contract BEFORE the membrane
                x0 = batch.x0.to(device)
                if self.preprocessor is not None:
                    x0 = self.preprocessor.transform(x0)   # membrane touches the generated quantity only
                # Conditioning rides to the model's device but is NEVER pushed through the membrane (Inv 9).
                cond = cond_to(batch.cond, device)
                if augment is not None:
                    cond = augment(cond, generator=aug_gen)     # main process => resumable (Inv 5)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.amp):
                    micro = self.method.loss(train_model, x0, cond=cond, generator=noise_gen)
                    micro = micro / self.grad_accum    # so summed grads equal one effective-batch step
                micro.backward()                       # grads ACCUMULATE across the micro-batches
                loss = micro.detach() if loss is None else loss + micro.detach()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()
            if lr_sched is not None:
                lr_sched.step()
            self.ema.update(self.model)

            self._last_losses.append(loss)   # already detached; defer the device→host sync out of the loop
            self._completed_steps = step
            if self.log_every and (step % self.log_every == 0 or step == start_step):
                lv = loss.item()   # one device→host sync on the log cadence (reused for both sinks)
                print(f"[train] step {step}/{self.steps}  loss={lv:.4f}")
                logger.log({"loss": lv, "lr": opt.param_groups[0]["lr"]}, step=step)
            # Held-out val pass (dedicated generators) → log + best-checkpoint. Off at val_every=0.
            if self.val_every and self._val_idx is not None and step % self.val_every == 0:
                vloss = self._val_loss(device, val_batch_gen, val_noise_gen)
                logger.log({"val_loss": vloss}, step=step)
                if self.save_best and self.ckpt_path and vloss < best_val:
                    best_val = vloss
                    self._save_best(vloss, step)
            # Full mid-training eval (sample+metrics via the dedicated sample_seed gen). Off at 0.
            if self.eval_every and step % self.eval_every == 0:
                logger.log(self.evaluate(), step=step)
                self.model.train()             # evaluate()->sample() leaves the model in eval()
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

    def _render(self, x: torch.Tensor) -> None:
        """Best-effort: hand generated samples to the configured visualizer (no-op when unset).
        Visualization must never fail a run, so errors are caught and reported."""
        viz = self.visualizer
        if viz is not None and hasattr(viz, "render"):
            out_dir = self._output_dir()
            if out_dir is not None and hasattr(viz, "out_dir"):
                viz.out_dir = out_dir           # land beside samples.npz, not a flat ./output
            try:
                viz.render(x)
            except Exception as e:
                print(f"[eval] visualizer skipped: {e}")

    def _output_dir(self) -> Optional[str]:
        """Where samples/metrics land, mirroring `ckpt_path` (`checkpoints/<...>.pt` →
        `output/<...>/`), else a sibling `<stem>/` dir. None when no `ckpt_path` (nothing persisted)."""
        if not self.ckpt_path:
            return None
        parts = Path(self.ckpt_path).with_suffix("").parts
        if "checkpoints" in parts:
            i = parts.index("checkpoints")
            return str(Path("output", *parts[i + 1:]))
        return str(Path(self.ckpt_path).with_suffix(""))

    def _ensure_split(self, n: int, device: torch.device) -> None:
        """Deterministic train/val index partition — built ONLY when val_frac>0, so val_frac=0
        constructs no generator and touches nothing (Invariant 5). Idempotent; also used lazily by
        `_val_batch` so `forge eval` gets a held-out batch without having run `train()`."""
        if self.val_frac <= 0 or self._val_idx is not None:
            return
        perm = torch.randperm(n, generator=make_generator(self.seed, device), device=device)
        n_val = max(1, int(n * self.val_frac))
        self._val_idx, self._train_idx = perm[:n_val], perm[n_val:]

    @torch.no_grad()
    def _val_loss(self, device: torch.device, batch_gen, noise_gen) -> float:
        """Held-out loss = method.loss on a val-pool batch, in eval mode. Dedicated generators only."""
        self.model.eval()
        sel = torch.randint(0, self._val_idx.numel(), (self.batch_size,), generator=batch_gen, device=device)
        batch = self.dataset.batch(self._val_idx[sel])
        x0 = batch.x0.to(device)
        if self.preprocessor is not None:
            x0 = self.preprocessor.transform(x0)
        cond = cond_to(batch.cond, device)          # dict conds too (Inv 9: never the membrane)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.amp):
            loss = self.method.loss(self.model, x0, cond=cond, generator=noise_gen)
        self.model.train()
        return float(loss)

    def _save_best(self, val_loss: float, step: int) -> None:
        """Save <ckpt>.best.pt + a sibling <ckpt>.best.metrics.json (the numbers that named it best)."""
        p = Path(self.ckpt_path)
        self.save_checkpoint(str(p.parent / (p.stem + ".best.pt")))
        save_metrics(str(p.parent), {"val_loss": val_loss}, step=step,
                     name=p.stem + ".best.metrics.json")

    def _val_batch(self):
        """Normalized held-out batch for data-driven metrics; None when no val split. Builds the
        split lazily so `forge eval` (no train()) still gets held-out data when val_frac>0. Uses a
        dedicated generator — never the train streams."""
        device = torch.device(self.device)
        self._ensure_split(self.dataset.num_items, device)
        if self._val_idx is None:
            return None
        gen = make_generator(self.sample_seed + 1, device)
        sel = torch.randint(0, self._val_idx.numel(), (self.batch_size,), generator=gen, device=device)
        x0 = self.dataset.batch(self._val_idx[sel]).x0.to(device)
        if self.preprocessor is not None:
            x0 = self.preprocessor.transform(x0)
        return x0

    def evaluate(self) -> dict:
        radius = self.eval_radius
        x = self.sample()                              # one draw, reused for render + persist + metrics
        self._render(x)
        held_out = self._val_batch()
        env = self.environment
        metrics: dict[str, float] = {"n": float(x.shape[0])}
        # Env-delegated metric (capability check, not a cont/disc branch) merges, no longer early-returns.
        if env is not None and hasattr(env, "evaluate"):
            metrics.update(env.evaluate(x))
        if self.metric is not None:
            metrics.update(self.metric(samples=x, held_out=held_out))
        elif env is not None and hasattr(env, "means"):   # back-compat fallback ONLY when no metric
            means = env.means.to(x.device)
            nearest = torch.cdist(x, means).min(dim=1).values
            metrics["mode_coverage"] = float((nearest < radius).float().mean().item())
            metrics["radius"] = float(radius)
        self._persist_eval(x, metrics)
        return metrics

    def _persist_eval(self, samples: torch.Tensor, metrics: dict) -> None:
        """Persist an eval draw + its step-stamped metrics. Shared with PlanningRunner.evaluate so
        both runners write the same artifacts the same way (the design contract: the runner persists
        samples.npz + step-stamped metrics.json). Change eval persistence here, not per override."""
        out_dir = self._output_dir()
        if out_dir is not None:
            save_samples(out_dir, samples)
            save_metrics(out_dir, metrics, step=self._completed_steps)

    # ── checkpoint ──────────────────────────────────────────────────────────────────────────────
    def _relocate_ema(self, device: torch.device) -> None:
        if self.ema is not None:
            self.ema.shadow = {k: v.to(device) for k, v in self.ema.shadow.items()}

    def _rng_state(self) -> Optional[dict]:
        if self._batch_gen is None or self._noise_gen is None:
            return None
        state = {
            "batch_gen": self._batch_gen.get_state().cpu(),
            "noise_gen": self._noise_gen.get_state().cpu(),
            "torch": torch.get_rng_state(),
        }
        if self._aug_gen is not None:
            state["aug_gen"] = self._aug_gen.get_state().cpu()   # rides the EXISTING rng_state dict
        return state

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

    # Config fields a strict state_dict load CANNOT catch: two checkpoints can agree on every tensor
    # SHAPE and still disagree on what the numbers MEAN. `output_type` reinterprets the same tensor
    # as ε vs x0; `schedule` swaps the noise curve under identical weights; `environment` decides
    # which task the plans are scored in — and locomotion siblings collide exactly here, since
    # halfcheetah and walker2d are both transition_dim 23 (act 6 + obs 17), so their checkpoints are
    # shape-identical and load into each other silently.
    # NOT checked, deliberately: `sampler` (sampling a DDPM-trained model with DDIM is a legitimate
    # choice, not a mismatch), `method` (training-only), and `preprocessor` (a kind mismatch already
    # fails loudly — its load_state_dict KeyErrors on the wrong stat keys).
    _CKPT_IDENTITY_KEYS = (
        ("model", "name"),
        ("model", "params", "output_type"),
        ("schedule", "name"),
        ("environment", "params", "name"),
    )

    @staticmethod
    def _dig(cfg, path):
        for key in path:
            if not hasattr(cfg, "get"):
                return None
            cfg = cfg.get(key)
        return cfg

    def _check_checkpoint_matches_config(self, ckpt: dict) -> None:
        """Fail loudly when a checkpoint disagrees with the experiment it is being loaded into.

        Invariant 5 records the config in every checkpoint; this reads it back. Only meaningful when
        a LIVE config exists to compare against: `resolved_config` is set solely by the CLI, so this
        fires on the risky `forge sample experiment=X` + `ckpt_path=Y` path and auto-no-ops for
        `from_checkpoint`, which builds FROM the checkpoint and is self-consistent by construction.
        Pre-config checkpoints (config=None) are skipped rather than rejected.
        """
        live, saved = self.resolved_config, ckpt.get("config")
        if not live or not saved:
            return
        bad = [
            f"  {'.'.join(path)}: experiment={a!r} but checkpoint={b!r}"
            for path in self._CKPT_IDENTITY_KEYS
            for a, b in [(self._dig(live, path), self._dig(saved, path))]
            # Only a genuine disagreement counts. A key present on one config but absent on the
            # other (schema drift) is unknowable, not a mismatch — comparing there false-rejects a
            # valid checkpoint. Both-present-and-differ is still caught.
            if a is not None and b is not None and a != b
        ]
        if bad:
            raise ValueError(
                "checkpoint does not match this experiment — the weights load (shapes agree) but "
                "would be interpreted under the wrong config:\n" + "\n".join(bad) +
                "\nPoint runner.params.ckpt_path at this experiment's own checkpoint, or run "
                "`forge sample checkpoint=<path.pt>` to rebuild from the checkpoint's own config."
            )

    def load_state(self, ckpt: dict) -> None:
        """Restore weights + EMA + preprocessor stats from a self-contained checkpoint."""
        self._check_checkpoint_matches_config(ckpt)
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
        # The step travels in the checkpoint (Invariant 5) and save_metrics stamps with it; without
        # this a sample-only run writes `"step": 0` and clobbers the training run's metrics.json.
        self._completed_steps = int(ckpt.get("step", 0))

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

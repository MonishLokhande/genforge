"""Autoregressive method + sampler — the two pieces genforge's diffusion built-ins don't cover.

`method=autoregressive` is next-token cross-entropy (mean nats/token, so it doubles as a per-token
likelihood — `nelbo=True` lets forge's `perplexity` metric consume it unchanged). `sampler=
autoregressive` generates left-to-right; it has no reverse `step` (generation happens in `sample`),
which is the honest shape of a non-diffusion guest paradigm. The model is the shared `transformer`
with `causal=true, time_conditioned=false`; nothing else in the stack knows AR exists.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from forge.core.interfaces import Method, Metric, Sampler
from forge.core.registry import register
from forge.core.types import SamplerOutput
from forge.utils.torch_utils import model_device


@register("method", "autoregressive")
class Autoregressive(Method):
    """Next-token cross-entropy. `loss` is mean nats/token, so `perplexity = exp(loss)`."""

    nelbo = True                                       # forge's Perplexity metric reads this loss as a likelihood

    def __init__(self, schedule=None, space=None, criterion=None):
        super().__init__(schedule, space, criterion)   # deps injected but unused (AR is not diffusion)

    def loss(self, model, x0: torch.Tensor, cond=None, generator=None) -> torch.Tensor:
        # x0: (B, L) int64. Shift by one: predict token i+1 from tokens ≤ i (causal model).
        logits = model(x0[:, :-1], None, cond)         # (B, L-1, V)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0[:, 1:].reshape(-1))


@register("sampler", "autoregressive")
class ARSampler(Sampler):
    """Left-to-right generation: seed one token, sample the rest with temperature / top-k."""

    def __init__(self, model, schedule=None, space=None, control=None, temperature: float = 1.0,
                 top_k: Optional[int] = None, start_token: int = 0, eos_id: Optional[int] = None):
        super().__init__(model, schedule, space, control)
        self.temperature = float(temperature)
        self.top_k = None if top_k in (None, 0) else int(top_k)
        self.start_token = int(start_token)
        self.eos_id = None if eos_id is None else int(eos_id)

    def step(self, x, t, s, cond=None):
        raise NotImplementedError("ARSampler generates in sample(), not via diffusion reverse steps.")

    @torch.no_grad()
    def sample(self, shape, n_steps=None, cond=None, generator=None, return_chain=False):
        n, length = int(shape[0]), int(shape[1])
        device = model_device(self.model)
        block = int(getattr(self.model, "length", length))     # max context the model was trained on
        seed = self.eos_id if self.eos_id is not None else self.start_token
        idx = torch.full((n, 1), seed, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(length - 1):
            logits = self.model(idx[:, -block:], None, None)[:, -1, :] / self.temperature
            if self.top_k is not None:
                v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            if self.eos_id is not None:                        # once a sequence ends, pad with EOS
                nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, self.eos_id), nxt)
                done = done | nxt.squeeze(1).eq(self.eos_id)
            idx = torch.cat([idx, nxt], dim=1)
        return SamplerOutput(samples=idx[:, :length])


@register("metric", "ar_perplexity")
class ARPerplexity(Metric):
    """HONEST held-out perplexity: scores the environment's DISJOINT val region (env.val_tokens),
    NOT the runner's index split — which leaks for overlapping LM windows. Log-domain average of the
    per-token CE, exponentiated once (averaging per-batch perplexity would be Jensen-biased high).

    Self-draws its data (ignores the runner's `held_out`), which is the only honest source; it must
    be paired with an env exposing `val_tokens` (ar_text) or it raises."""

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 block_size: int = 64, n_batches: int = 50, batch_size: int = 32, seed: int = 1234):
        super().__init__(environment, model, method, dataset, schedule)
        self.block_size = int(block_size)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __call__(self, samples=None, held_out=None) -> dict:
        val = getattr(self.environment, "val_tokens", None)
        if val is None:
            raise ValueError(
                "ar_perplexity needs an environment exposing val_tokens (the disjoint held-out "
                "region), e.g. environment=ar_text with val_frac>0."
            )
        seq_len = self.block_size + 1
        if val.numel() <= seq_len:
            raise ValueError(
                f"val region ({val.numel()} tokens) < one window ({seq_len}); raise "
                f"environment.params.val_frac or lower this metric's block_size."
            )
        dev = next(self.model.parameters()).device
        g = torch.Generator().manual_seed(self.seed)          # CPU gen; indexes the CPU corpus
        offsets = torch.arange(seq_len)
        was_training = self.model.training
        self.model.eval()
        nats = 0.0
        with torch.no_grad():
            for _ in range(self.n_batches):
                idx = torch.randint(0, val.numel() - seq_len + 1, (self.batch_size,), generator=g)
                batch = val[idx.reshape(-1, 1) + offsets.reshape(1, -1)].to(torch.int64).to(dev)
                nats += float(self.method.loss(self.model, batch)) / self.n_batches
        if was_training:
            self.model.train()
        return {"val_perplexity": math.exp(nats), "val_nats_per_token": nats}

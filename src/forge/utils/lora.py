"""LoRA: low-rank adapters for parameter-efficient fine-tuning (torch-only, no `peft`).

A single generic injector replaces matching ``nn.Linear`` modules with a :class:`LoRALinear`
that freezes the base weight and trains only a low-rank ``B @ A`` update. It walks
``model.named_modules()`` by the child's *leaf* name, so it works on **any** ``nn.Module`` —
the genforge-native ``Transformer`` (attention ``q``/``k``/``v``/``out_proj`` + FFN
``linear1``/``linear2`` + ``head``) and HuggingFace backbones (``query``/``key``/``value``/
``dense``) alike. The native Transformer writes attention out with explicit ``nn.Linear`` q/k/v/out
(not ``nn.MultiheadAttention``), so the whole block is reachable — full native coverage.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Default targets for the genforge-native Transformer (verified against named_modules()):
# full attention (q/k/v/out_proj) + FFN (linear1/linear2) + output head.
DEFAULT_TARGETS: tuple[str, ...] = ("q", "k", "v", "out_proj", "linear1", "linear2", "head")


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a low-rank update: ``y = base(x) + (x A^T) B^T * s``.

    ``A`` is kaiming-initialized and ``B`` is zero, so the adapter is a **no-op at init** — the
    wrapped module is bit-identical to the base until training moves ``B`` (clean fine-tune start).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> int:
    """Replace every ``nn.Linear`` whose *leaf* name is in ``targets`` with a :class:`LoRALinear`,
    then freeze all non-adapter parameters. Returns the number of layers wrapped.

    Raises ``ValueError`` when nothing matches (a misconfigured ``targets`` would otherwise
    silently leave every parameter trainable — the optimizer's ``requires_grad`` filter sees no
    frozen params — and you'd train the full model thinking LoRA fired). Never warn-and-continue.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0, got {r}.")
    alpha = float(alpha) if alpha is not None else 2.0 * r
    targets = set(targets)

    # Collect first, then mutate — replacing children mid-iteration over modules() is unsafe.
    hits = [
        (module, name, child)
        for module in model.modules()
        for name, child in module.named_children()
        if isinstance(child, nn.Linear) and name in targets
    ]
    if not hits:
        avail = sorted(
            {n for m in model.modules() for n, c in m.named_children() if isinstance(c, nn.Linear)}
        )
        raise ValueError(
            f"apply_lora: targets={sorted(targets)} matched no nn.Linear. "
            f"Available leaf Linear names: {avail}"
        )

    keep: set[int] = set()
    for module, name, child in hits:
        lora = LoRALinear(child, r, alpha, dropout)
        setattr(module, name, lora)
        keep.update((id(lora.A), id(lora.B)))

    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    return len(hits)


if __name__ == "__main__":
    # Self-check: no-op at init, only adapters trainable, a grad step changes the output.
    torch.manual_seed(0)
    net = nn.Sequential()
    net.add_module("linear1", nn.Linear(8, 16))
    net.add_module("act", nn.ReLU())
    net.add_module("head", nn.Linear(16, 4))
    x = torch.randn(3, 8)
    before = net(x).clone()

    n = apply_lora(net, r=4, alpha=8, targets=("linear1", "head"))
    assert n == 2, n
    assert torch.allclose(net(x), before, atol=1e-6), "B-zero init must be a no-op"

    trainable = [p for p in net.parameters() if p.requires_grad]
    frozen = [p for p in net.parameters() if not p.requires_grad]
    assert len(trainable) == 4 and len(frozen) > 0, (len(trainable), len(frozen))  # A,B per layer

    out = net(x).sum()
    out.backward()
    assert all(p.grad is not None for p in trainable), "adapters must receive grad"
    assert all(p.grad is None for p in frozen), "frozen base must get no grad"
    with torch.no_grad():
        for p in trainable:
            p += 0.1 * p.grad
    assert not torch.allclose(net(x), before, atol=1e-6), "a step must change the output"

    try:
        apply_lora(nn.Linear(2, 2), targets=("nonexistent",))
        raise AssertionError("expected ValueError on zero matches")
    except ValueError:
        pass
    print("lora self-check OK")

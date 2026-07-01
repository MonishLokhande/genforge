"""fbsde — learn a value function by minimizing the (deterministic) HJB/BSDE residual.

The EDL FBSDE template on a checkable linear-quadratic toy: dynamics ``dx = u dt``, running cost
``½(q x² + u²)``. The stationary HJB after the optimal ``u* = −V'(x)`` is
``0 = ½ q x² − ½ V'(x)²`` (the σ=0 / Hamilton–Jacobi reduction), whose convex solution is the
Riccati value ``V(x) = ½√q x²`` (so ``V'(x)² = q x²``). The method minimizes the residual over
collocation points and anchors ``V(0)=0``; the paired `FBSDEControl` consumes the learned V via a
checkpoint (Invariant 6).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.interfaces import Method, Model
from ..core.registry import register


@register("method", "fbsde")
class FBSDE(Method):
    def __init__(self, schedule, space, q: float = 4.0):
        super().__init__(schedule, space)
        self.q = float(q)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,                                    # collocation points (B, dim)
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        x = x0.detach().requires_grad_(True)
        v = model(x)                                         # (B, 1)
        (grad,) = torch.autograd.grad(v.sum(), x, create_graph=True)
        gx2 = (grad**2).sum(dim=-1, keepdim=True)
        xx2 = (x**2).sum(dim=-1, keepdim=True)
        residual = 0.5 * self.q * xx2 - 0.5 * gx2            # HJB residual (→ 0)
        anchor = (model(torch.zeros(1, x.shape[-1], device=x.device)) ** 2).mean()
        # V'(x)² = q x² admits ±V and per-side sign flips. The LQ value is even and non-negative,
        # so symmetry V(x)=V(−x) + positivity pins the convex Riccati branch V = ½√q x².
        positivity = torch.relu(-v).mean()
        symmetry = ((v - model(-x.detach())) ** 2).mean()
        return (
            (residual**2).mean()
            + anchor
            + symmetry
            + 0.1 * positivity
        )

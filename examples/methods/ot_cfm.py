"""Minibatch optimal-transport conditional flow matching (OT-CFM).

Identical to flow matching except the prior noise ``ε`` is **coupled** to the data within each
minibatch by an optimal assignment (minimizing Σ‖x_data − ε‖²). The straighter couplings make the
learned velocity field straighter, improving few-step ODE sampling. Lives behind the ``flow`` extra
(needs SciPy for the assignment).
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core.interfaces import Method, Model
from forge.core.registry import register


def _ot_permutation(x_data: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Return indices that permute ``eps`` to its squared-distance-optimal assignment to ``x_data``."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only without the flow extra
        raise ModuleNotFoundError(
            "ot_cfm needs SciPy (the `flow` extra): `uv sync --extra flow`."
        ) from e

    cost = torch.cdist(x_data, eps) ** 2
    _, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(col, device=eps.device, dtype=torch.long)


@register("method", "ot_cfm")
class OTCFM(Method):
    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.t_eps = float(t_eps)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        # Draw prior noise and couple it to the data by minibatch OT.
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        perm = _ot_permutation(x0, eps)
        eps = eps[perm]

        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps)
        a = self.schedule.alpha(t).reshape(-1, *([1] * (x0.ndim - 1)))
        s = self.schedule.sigma(t).reshape(-1, *([1] * (x0.ndim - 1)))
        xt = a * x0 + s * eps
        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        w = self.schedule.loss_weight(model.output_type, t).reshape(-1, *([1] * (x0.ndim - 1)))
        return torch.mean(w * (pred - target) ** 2)

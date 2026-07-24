"""value_training — amortize a reward into a value model V(x) (regression to the reward).

The training counterpart of `ValueGuidance` (Invariant 6): it produces an artifact (a value-model
checkpoint); the controller consumes that artifact through the checkpoint and nowhere else. Here the
reward is a quadratic bowl ``−w‖x − target‖²`` defined by the method's own params (kept out of the
build-order coupling), so V learns a smooth landscape whose ∇ points toward the target.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from forge.core.interfaces import Method, Model
from forge.core.registry import register


@register("method", "value_training")
class ValueTraining(Method):
    def __init__(self, schedule, space, target: Sequence[float] = (2.0, 0.0), weight: float = 1.0):
        super().__init__(schedule, space)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.weight = float(weight)

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.weight * ((x - self.target.to(x.device)) ** 2).sum(dim=-1, keepdim=True)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        target = self.reward(x0)
        # Pass t EXPLICITLY: the Model contract is forward(x, t, cond=None). Calling model(x0) only
        # worked for value models whose `t` happens to be optional (ValueMLP); a value model with the
        # contract's required positional `t` (ValueUNet) raised TypeError, so this method could not
        # train the locomotion value head at all. t=0 is the honest value here — V is regressed on
        # CLEAN x0 (no forward diffusion), so the learned V is time-independent. NOTE: that is a
        # deliberate simplification of Janner's value function, which is trained on NOISED x_t at a
        # sampled t; a faithful port would diffuse x0 here and regress V(x_t, t).
        t = torch.zeros(x0.shape[0], device=x0.device, dtype=torch.float32)
        pred = model(x0, t).reshape(target.shape)
        return ((pred - target) ** 2).mean()

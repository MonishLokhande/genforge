"""Vision-conditioned trajectory denoiser — forge port of the reference implementation's ``temporal_unet_image``,
checkpoint-compatible with the trained pusht/robomimic image policies.

Per-camera ResNet18 (GroupNorm swap, dp recipe; stock torchvision) feeds the global-cond
path of an inner ``TemporalUNetJanner`` (forge's port of the same conditional U-Net the
checkpoints' ``unet.*`` keys came from — 148-key strict match, proven on can_ph).

Rolls out end-to-end: the robomimic adapter serves camera frames (`image_keys`), `MultiStepWrapper`
keeps a parallel frame deque, and `PolicyWrapper` hands this model a dict cond. Proprio arrives RAW
and is normalized by `obs_normalizer` below (Inv 9 — the membrane only ever touches x = actions).

# ⚠ CHECKPOINT PARITY — module/parameter names frozen (`encoder.N.*`, `unet.*`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from forge.core.interfaces import Model
from forge.core.registry import register
from examples.models.temporal_unet_janner import TemporalUNetJanner


def resnet18_groupnorm(feat_dim: int, groups: int = 16) -> nn.Module:
    """torchvision ResNet18, BatchNorm→GroupNorm, random init, head → feat_dim."""
    import torchvision

    net = torchvision.models.resnet18(
        weights=None, norm_layer=lambda c: nn.GroupNorm(groups, c))
    net.fc = nn.Linear(net.fc.in_features, feat_dim)
    return net


class MinMaxObsNormalizer(nn.Module):
    """Per-dimension min-max → [-1, 1] for the proprio conditioning stream.

    Invariant 9: obs normalization is the MODEL's job, never the membrane preprocessor — proprio
    rides in `cond`, and the membrane only ever touches the generated quantity x.

    MIN-MAX, deliberately: the trained image checkpoints normalized proprio with a
    LimitsNormalizer (``2*(x-min)/range - 1``), which is the same affine map as forge's `MinMax`
    preprocessor. The mean/std `ObsNormalizer` in models/temporal_unet_janner.py is DIFFERENT math
    and would silently degrade rollout rather than error — do not substitute it.

    Buffers are PERSISTENT so they travel in state_dict (Invariant 5: sample from a .pt alone).
    """

    def __init__(self, obs_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("min", torch.zeros(obs_dim))
        self.register_buffer("max", torch.ones(obs_dim))

    def fit(self, obs: Tensor) -> None:
        """Compute per-dim min/max from obs ``(..., obs_dim)`` and store as persistent buffers."""
        obs = obs.reshape(-1, obs.shape[-1])
        self.min.copy_(obs.min(dim=0).values.to(self.min.dtype))
        self.max.copy_(obs.max(dim=0).values.to(self.max.dtype))

    def forward(self, x: Tensor) -> Tensor:
        rng = (self.max - self.min).clamp_min(self.eps)
        return 2.0 * (x - self.min) / rng - 1.0


# Registered under BOTH names: `vision_unet` is the forge name, while `temporal_unet_image` is the
# name embedded in the trained pusht/robomimic image checkpoints — the alias lets those `.pt` files
# reconstruct self-contained (`forge sample checkpoint=…image_ddpm.pt`), which is what makes
# Invariant 5's "sample from a .pt alone" hold for a transplanted image policy.
@register("model", "temporal_unet_image")
@register("model", "vision_unet")
class VisionUNet(Model):
    """ResNet18 visual encoder(s) + conditional temporal U-Net (joint-trained).

    forward(x, t, cond) where cond is a dict with ``obs_images`` (B, To, n_cam, C, H, W)
    and, when proprio_dim > 0, ``obs_history`` (B, ≥To, proprio_dim).
    """

    is_image = True
    output_type: str

    def __init__(
        self,
        *,
        horizon: int,
        transition_dim: int,
        n_obs_steps: int,
        n_cam: int,
        image_feat_dim: int = 64,
        proprio_dim: int = 0,
        share_encoder: bool = False,
        dim: int = 256,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        cond_predict_scale: bool = True,
        groups: int = 16,
        output_type: str = "eps",
    ) -> None:
        super().__init__()
        self.output_type = output_type
        self.n_obs_steps = int(n_obs_steps)
        self.n_cam = int(n_cam)
        self.image_feat_dim = int(image_feat_dim)
        self.proprio_dim = int(proprio_dim)
        self.share_encoder = bool(share_encoder)

        n_enc = 1 if share_encoder else self.n_cam
        self.encoder = nn.ModuleList(
            [resnet18_groupnorm(self.image_feat_dim, groups) for _ in range(n_enc)])

        # Fitted by TrainingRunner._fit_model_conditioning (duck-typed on `obs_normalizer` +
        # dataset.cond_fit_tensor) — no runner branching. None when there is no proprio stream,
        # which also makes that hook a no-op.
        self.obs_normalizer = MinMaxObsNormalizer(self.proprio_dim) if self.proprio_dim > 0 else None

        self.cond_dim = self.n_obs_steps * (self.n_cam * self.image_feat_dim + self.proprio_dim)
        self.unet = TemporalUNetJanner(
            horizon=horizon, transition_dim=transition_dim, cond_dim=self.cond_dim,
            dim=dim, dim_mults=dim_mults, cond_predict_scale=cond_predict_scale,
            output_type=output_type, obs_dim=0)

    def encode_images(self, images: Tensor) -> Tensor:
        """``(B, To, n_cam, C, H, W)`` uint8/float → ``(B, To, n_cam·feat)``."""
        b, to, n_cam = images.shape[:3]
        x = images.float()
        if x.max() > 1.5:                         # uint8 → [0, 1] (no-op if already scaled)
            x = x / 255.0
        feats = []
        for cam in range(n_cam):
            enc = self.encoder[0] if self.share_encoder else self.encoder[cam]
            f = enc(x[:, :, cam].reshape(b * to, *x.shape[3:]))   # (B·To, feat)
            feats.append(f.reshape(b, to, -1))
        return torch.cat(feats, dim=-1)            # (B, To, n_cam·feat)

    def forward(self, x: Tensor, t: Tensor, cond=None) -> Tensor:
        if not isinstance(cond, dict) or "obs_images" not in cond:
            raise ValueError("vision_unet requires cond['obs_images']")
        g = self.encode_images(cond["obs_images"])                # (B, To, n_cam·feat)
        if self.proprio_dim > 0:
            proprio = cond.get("obs_history")
            if proprio is None:
                raise ValueError(f"proprio_dim={self.proprio_dim} but no obs_history given")
            proprio = proprio[:, : self.n_obs_steps].to(g.dtype)
            # RAW proprio in, normalized here (Inv 9) — cond never crosses the membrane, so the
            # train-time and rollout-time streams are both raw and both normalized by these buffers.
            proprio = self.obs_normalizer(proprio)
            g = torch.cat([g, proprio], dim=-1)
        global_cond = g.flatten(1)                                # (B, cond_dim)
        return self.unet(x, t, global_cond)

"""Vision experiment family — colocates the image-observation trajectory-UNet plugin
(forge port of the reference implementation's ``temporal_unet_image``) under experiment/, mirroring
experiment/robotics/locomotion.

Image observations are wired end-to-end: the robomimic adapter yields camera frames when
``image_keys`` is set, `TrajectoryWindowDataset` carries them as a uint8 flat stream and emits a
dict cond, and `MultiStepWrapper`/`PolicyWrapper` supply the same dict at rollout. See the
``can_image_ddpm`` leaf; ``forge sample checkpoint=<image_ddpm.pt>`` also works standalone.
"""
from . import vision_unet  # noqa: F401 — @register("model", "temporal_unet_image" / "vision_unet")

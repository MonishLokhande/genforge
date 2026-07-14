"""Locomotion experiment family — colocates the bespoke plugin CODE with the Hydra configs in this
same folder (attention_unet / value_unet / guided_planning). Declaring
``plugins: [experiment.robotics.locomotion]`` in a leaf imports this package, firing the
``@register`` decorators below. (The matching ``/model`` config stubs stay in the shared top-level
``model/`` group, alongside genforge's builtins.)
"""
from . import attention_unet  # noqa: F401 — @register("model", "attention_unet")
from . import value_unet  # noqa: F401 — @register("model", "value_unet")
from . import guided_planning  # noqa: F401 — @register("control", "diffuser_value_guidance")

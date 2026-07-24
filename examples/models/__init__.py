"""Out-of-tree models example plugins — importing fires their @register decorators."""
from . import categorical  # noqa: F401
from . import temporal_unet  # noqa: F401
from . import temporal_unet_janner  # noqa: F401
from . import transformer  # noqa: F401
from . import value  # noqa: F401

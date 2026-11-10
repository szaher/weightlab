"""WeightLab: local-first, CPU-only LLM checkpoint explorer.

Supports strict SafeTensors reads, exact ROI values, local heatmaps, and
checkpoint/LoRA change reports. The graphical comparison UI is still planned.
"""

from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.formats.base import Consistency, TensorMeta
from weightlab.formats.checkpoint import Checkpoint, open_checkpoint

__version__ = "0.1.0a1"
PARSER_VERSION = 1

__all__ = [
    "PARSER_VERSION",
    "Checkpoint",
    "Consistency",
    "ErrorCode",
    "TensorMeta",
    "WeightLabError",
    "__version__",
    "open_checkpoint",
]

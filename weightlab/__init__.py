"""WeightLab: local-first, CPU-only LLM checkpoint explorer.

Phase 0 surface: a strict, read-only SafeTensors reader with sharded index validation and
exact, lossless region-of-interest (ROI) reads. No tiles, API or UI yet.
"""

from weightlab.core.errors import ErrorCode, WeightLabError
from weightlab.formats.base import Consistency, TensorMeta
from weightlab.formats.checkpoint import Checkpoint, open_checkpoint

__version__ = "0.0.1.dev0"
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

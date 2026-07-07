"""Optional diffusion components for MolScribe.

The existing graph predictor remains the baseline.  These modules provide
isolated building blocks for experimental discrete edge diffusion decoders.
"""

from .edge_utils import EDGE_IGNORE_INDEX, EDGE_MASK_ID, EDGE_NUM_CLASSES

__all__ = [
    "EDGE_IGNORE_INDEX",
    "EDGE_MASK_ID",
    "EDGE_NUM_CLASSES",
]

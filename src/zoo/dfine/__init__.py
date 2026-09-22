"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The D-FINE model: ``DFINE`` wires a backbone, the ``HybridEncoder`` and the ``DFINETransformer``
decoder; ``DFINECriterion`` and ``HungarianMatcher`` train it and ``DFINEPostProcessor`` turns its
outputs into detections. Importing the package registers
all of them for the configs.
"""

from ...solver.matcher import HungarianMatcher
from .dfine import DFINE
from .dfine_criterion import DFINECriterion
from .dfine_decoder import DFINETransformer
from .hybrid_encoder import HybridEncoder
from .postprocessor import DFINEPostProcessor

__all__ = [
    "DFINETransformer",
    "DFINE",
    "DFINECriterion",
    "DFINEPostProcessor",
    "HungarianMatcher",
    "HybridEncoder",
]

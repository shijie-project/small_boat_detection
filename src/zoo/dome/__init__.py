"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The D-FINE model: ``DOME`` wires a backbone, the ``HybridEncoder`` and the ``DFINETransformer``
decoder; ``DomeCriterion`` and ``HungarianMatcher`` train it and ``DomePostProcessor`` turns its
outputs into detections. Importing the package registers
all of them for the configs.
"""

from ...solver.matcher import HungarianMatcher
from .dfine_decoder import DFINETransformer
from .dome import DOME
from .dome_criterion import DomeCriterion
from .hybrid_encoder import HybridEncoder
from .postprocessor import DomePostProcessor

__all__ = [
    "DFINETransformer",
    "DOME",
    "DomeCriterion",
    "DomePostProcessor",
    "HungarianMatcher",
    "HybridEncoder",
]

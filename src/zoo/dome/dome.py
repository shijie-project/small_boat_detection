"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register

__all__ = ["DOME"]


@register()
class DOME(nn.Module):
    """
    Backbone -> encoder -> decoder. The encoder also receives the input image and the targets
    and hands the decoder a dict (``feats`` plus ``img_inputs``) rather than a plain feature list.
    """

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module = None, decoder: nn.Module = None):
        super().__init__()
        if encoder is None or decoder is None:
            raise ValueError(
                "DOME needs an encoder and a decoder: set `DOME: {encoder: ..., decoder: ...}` in the model "
                "config (HybridEncoder + DFINETransformer)"
            )
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    @property
    def wiring(self) -> str:
        return (
            f"{type(self).__name__}(backbone={type(self.backbone).__name__}, "
            f"encoder={type(self.encoder).__name__}, decoder={type(self.decoder).__name__})"
        )

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        encoder_out = self.encoder(feats, x, targets)
        return self.decoder(encoder_out, targets)

    def deploy(self):
        """Switch to inference: eval mode, with every module that can fold its layers folded."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

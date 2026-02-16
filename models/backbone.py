"""Encoder backbones for GMLFNet.

Supports:
- Res2Net-50 (v1b, 26w, 4s): proven in polyp segmentation (PraNet, SINet)
- PVTv2-B2: transformer-based, used by Polyp-PVT and newer methods

Both produce 4 multi-scale feature maps at strides [4, 8, 16, 32].
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class Res2NetBackbone(nn.Module):
    """Res2Net-50 (v1b, 26w, 4s) encoder.

    Produces 4 feature levels:
        - f1: stride 4,  channels 256
        - f2: stride 8,  channels 512
        - f3: stride 16, channels 1024
        - f4: stride 32, channels 2048

    Uses timm library for pretrained weights.
    """

    out_channels = [256, 512, 1024, 2048]

    def __init__(self, pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Res2Net backbone. Install: pip install timm")

        # Load Res2Net-50 from timm
        backbone = timm.create_model(
            "res2net50_26w_4s",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.backbone = backbone

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: input tensor (B, 3, H, W)

        Returns:
            list of 4 feature maps at increasing strides
        """
        features = self.backbone(x)
        return features


class PVTv2B2Backbone(nn.Module):
    """PVTv2-B2 transformer encoder.

    Produces 4 feature levels:
        - f1: stride 4,  channels 64
        - f2: stride 8,  channels 128
        - f3: stride 16, channels 320
        - f4: stride 32, channels 512

    Uses timm library for pretrained weights.
    """

    out_channels = [64, 128, 320, 512]

    def __init__(self, pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for PVTv2 backbone. Install: pip install timm")

        backbone = timm.create_model(
            "pvt_v2_b2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.backbone = backbone

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: input tensor (B, 3, H, W)

        Returns:
            list of 4 feature maps at increasing strides
        """
        features = self.backbone(x)
        return features


def get_backbone(name="res2net50", pretrained=True):
    """Factory function to create a backbone.

    Args:
        name: "res2net50" or "pvt_v2_b2"
        pretrained: whether to load pretrained ImageNet weights

    Returns:
        backbone module with .out_channels attribute
    """
    if name == "res2net50":
        return Res2NetBackbone(pretrained=pretrained)
    elif name == "pvt_v2_b2":
        return PVTv2B2Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose 'res2net50' or 'pvt_v2_b2'")

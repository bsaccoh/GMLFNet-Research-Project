"""Multi-scale decoder with Receptive Field Blocks and Reverse Attention.

Inspired by PraNet's decoder architecture:
- RFB modules enhance multi-scale receptive fields
- Partial decoder aggregates high-level features
- Reverse attention refines boundaries by erasing confident regions

The decoder accepts optional FiLM modulation parameters from the FAW module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFB(nn.Module):
    """Receptive Field Block for multi-scale feature enhancement.

    Uses parallel dilated convolutions with different rates to capture
    multi-scale context, then fuses via 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        cat = torch.cat([x0, x1, x2, x3], dim=1)
        out = self.conv_cat(cat) + self.conv_res(x)
        return F.relu(out, inplace=True)


class PartialDecoder(nn.Module):
    """Aggregates high-level features (f3, f4) for initial prediction."""

    def __init__(self, channel):
        super().__init__()
        self.conv_upsample1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(3 * channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(channel, 1, 1)

    def forward(self, f2, f3, f4):
        """Fuse f2, f3, f4 into an initial segmentation map.

        All inputs should have the same channel dimension.
        f3 and f4 are upsampled to match f2's spatial size.
        """
        f4_up = F.interpolate(f4, size=f2.shape[2:], mode="bilinear", align_corners=False)
        f4_up = self.conv_upsample1(f4_up)

        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        f3_up = self.conv_upsample2(f3_up)

        cat = torch.cat([f2, f3_up, f4_up], dim=1)
        fused = self.conv_concat(cat)
        out = self.conv_out(fused)
        return out, fused


class ReverseAttention(nn.Module):
    """Reverse Attention module.

    Erases already-predicted (confident) regions from feature maps,
    forcing the network to refine uncertain boundary areas.
    """

    def __init__(self, in_channels, channel):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(channel, 1, 1)

    def forward(self, x, prev_pred):
        """Apply reverse attention.

        Args:
            x: encoder feature map (B, C_in, H, W)
            prev_pred: previous prediction logits, upsampled to match x

        Returns:
            refined prediction logits (B, 1, H, W)
        """
        # Erase confident regions
        reverse_mask = 1 - torch.sigmoid(prev_pred)
        x = self.conv_input(x)
        x = x * reverse_mask
        x = self.conv_refine(x)
        out = self.conv_out(x)
        return out


class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with RFB, Partial Decoder, and Reverse Attention.

    Supports optional FiLM modulation from the FAW module.

    Args:
        encoder_channels: list of 4 channel sizes from encoder
        decoder_channel: unified channel size for decoder operations
    """

    def __init__(self, encoder_channels, decoder_channel=32):
        super().__init__()
        # RFB modules to reduce encoder features to unified channel
        self.rfb2 = RFB(encoder_channels[1], decoder_channel)
        self.rfb3 = RFB(encoder_channels[2], decoder_channel)
        self.rfb4 = RFB(encoder_channels[3], decoder_channel)

        # Partial decoder for initial prediction from high-level features
        self.partial_decoder = PartialDecoder(decoder_channel)

        # Reverse attention modules for progressive refinement
        self.ra4 = ReverseAttention(encoder_channels[3], decoder_channel)
        self.ra3 = ReverseAttention(encoder_channels[2], decoder_channel)
        self.ra2 = ReverseAttention(encoder_channels[1], decoder_channel)

    def forward(self, features, modulations=None):
        """Decode multi-scale features into segmentation maps.

        Args:
            features: list of 4 feature maps [f1, f2, f3, f4] from encoder
            modulations: optional list of (gamma, beta) from FAW module.
                        If provided, applies FiLM modulation after each RFB.

        Returns:
            main_pred: primary segmentation logits (B, 1, H_input, W_input)
            side_preds: list of side output logits for deep supervision
        """
        f1, f2, f3, f4 = features
        input_size = f1.shape[2] * 4, f1.shape[3] * 4  # Original input size

        # Apply RFB to reduce channels
        x2 = self.rfb2(f2)
        x3 = self.rfb3(f3)
        x4 = self.rfb4(f4)

        # Apply FiLM modulation if provided
        if modulations is not None:
            # modulations[0]->f2, modulations[1]->f3, modulations[2]->f4
            if len(modulations) >= 3:
                gamma2, beta2 = modulations[0]
                gamma3, beta3 = modulations[1]
                gamma4, beta4 = modulations[2]
                x2 = gamma2 * x2 + beta2
                x3 = gamma3 * x3 + beta3
                x4 = gamma4 * x4 + beta4

        # Partial decoder: initial prediction from high-level features
        pred_init, fused = self.partial_decoder(x2, x3, x4)

        # Upsample initial prediction to original size
        side_preds = []
        pred5 = F.interpolate(pred_init, size=input_size, mode="bilinear", align_corners=False)
        side_preds.append(pred5)

        # Reverse attention on f4
        pred4 = self.ra4(f4, F.interpolate(pred_init, size=f4.shape[2:], mode="bilinear", align_corners=False))
        pred4_up = F.interpolate(pred4, size=input_size, mode="bilinear", align_corners=False)
        side_preds.append(pred4_up)

        # Reverse attention on f3
        pred3 = self.ra3(f3, F.interpolate(pred4, size=f3.shape[2:], mode="bilinear", align_corners=False))
        pred3_up = F.interpolate(pred3, size=input_size, mode="bilinear", align_corners=False)
        side_preds.append(pred3_up)

        # Reverse attention on f2
        pred2 = self.ra2(f2, F.interpolate(pred3, size=f2.shape[2:], mode="bilinear", align_corners=False))
        main_pred = F.interpolate(pred2, size=input_size, mode="bilinear", align_corners=False)

        return main_pred, side_preds

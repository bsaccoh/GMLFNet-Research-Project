"""GMLFNet: Gradient-Based Meta-Learning with Fast Adaptation Weights
for Robust Multi-Centre Polyp Segmentation.

Full architecture:
    Encoder (Res2Net-50 or PVTv2-B2)
    -> Fast Adaptation Weights (FiLM modulation)
    -> Multi-Scale Decoder (RFB + Reverse Attention)

During meta-learning:
    - Inner loop: adapts FAW parameters (lightweight, ~100K params)
    - Outer loop: updates all parameters (encoder + FAW + decoder)
"""

import torch
import torch.nn as nn

from .backbone import get_backbone
from .decoder import MultiScaleDecoder
from .fast_adapt_weights import FastAdaptationWeights


class GMLFNet(nn.Module):
    """GMLFNet segmentation model.

    Args:
        backbone_name: "res2net50" or "pvt_v2_b2"
        decoder_channel: unified channel size for decoder (default 32)
        faw_hidden_dim: hidden dimension of FAW MLP
        faw_num_layers: number of MLP layers in FAW
        pretrained: load pretrained backbone weights
        use_faw: whether to use FAW module (set False for ablation)
    """

    def __init__(
        self,
        backbone_name="res2net50",
        decoder_channel=32,
        faw_hidden_dim=64,
        faw_num_layers=2,
        pretrained=True,
        use_faw=True,
    ):
        super().__init__()
        self.use_faw = use_faw

        # Encoder
        self.encoder = get_backbone(backbone_name, pretrained=pretrained)
        enc_channels = self.encoder.out_channels

        # Fast Adaptation Weights
        if use_faw:
            self.faw = FastAdaptationWeights(
                encoder_channels=enc_channels,
                num_modulation_layers=3,
                modulation_channels=decoder_channel,
                hidden_dim=faw_hidden_dim,
                num_layers=faw_num_layers,
            )
        else:
            self.faw = None

        # Decoder
        self.decoder = MultiScaleDecoder(
            encoder_channels=enc_channels,
            decoder_channel=decoder_channel,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: input images (B, 3, H, W)

        Returns:
            main_pred: segmentation logits (B, 1, H, W)
            side_preds: list of side output logits for deep supervision
        """
        # Encode
        features = self.encoder(x)

        # Generate modulation (if FAW enabled)
        modulations = None
        if self.use_faw and self.faw is not None:
            modulations = self.faw(features)

        # Decode
        main_pred, side_preds = self.decoder(features, modulations=modulations)

        return main_pred, side_preds

    def get_faw_parameters(self):
        """Return only FAW parameters (for selective inner-loop adaptation)."""
        if self.faw is not None:
            return list(self.faw.parameters())
        return []

    def get_non_faw_parameters(self):
        """Return encoder + decoder parameters (non-FAW)."""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        return params

    def freeze_non_faw(self):
        """Freeze encoder and decoder, leaving only FAW trainable.
        Used during MAML inner loop for selective adaptation.
        """
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for param in self.decoder.parameters():
            param.requires_grad_(False)
        if self.faw is not None:
            for param in self.faw.parameters():
                param.requires_grad_(True)

    def unfreeze_all(self):
        """Unfreeze all parameters. Used for outer loop."""
        for param in self.parameters():
            param.requires_grad_(True)

    def print_param_summary(self):
        """Print parameter count summary."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        faw_params = self.faw.get_param_count() if self.faw else 0
        total = enc_params + dec_params + faw_params

        print(f"Parameter Summary:")
        print(f"  Encoder:  {enc_params:>10,}")
        print(f"  Decoder:  {dec_params:>10,}")
        print(f"  FAW:      {faw_params:>10,}")
        print(f"  Total:    {total:>10,}")
        print(f"  FAW ratio: {faw_params/total*100:.2f}% of total")


def build_model(cfg):
    """Build GMLFNet from config.

    Args:
        cfg: Config object with model section

    Returns:
        GMLFNet instance
    """
    model = GMLFNet(
        backbone_name=cfg.model.backbone,
        decoder_channel=cfg.model.decoder_channels[-1] if hasattr(cfg.model, "decoder_channels") else 32,
        faw_hidden_dim=cfg.model.faw_hidden_dim,
        faw_num_layers=cfg.model.faw_num_layers,
        pretrained=True,
        use_faw=True,
    )
    return model

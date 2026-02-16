"""Loss functions for polyp segmentation.

- StructureLoss: weighted BCE + weighted IoU (from PraNet)
- BCEDiceLoss: standard BCE + Dice combination
- GMLFNetLoss: composite loss with deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureLoss(nn.Module):
    """Structure-aware loss from PraNet.

    Uses edge-distance weighting to emphasize boundary regions:
        weight = 1 + 5 * |AvgPool(mask) - mask|

    Combines weighted BCE and weighted IoU losses.
    """

    def forward(self, pred, mask):
        """
        Args:
            pred: prediction logits (B, 1, H, W) — before sigmoid
            mask: ground truth (B, 1, H, W) — binary {0, 1}
        """
        # Edge-distance weighting
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        # Weighted BCE
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        Args:
            pred: prediction logits (B, 1, H, W)
            mask: ground truth (B, 1, H, W)
        """
        bce = F.binary_cross_entropy_with_logits(pred, mask)

        pred_sigmoid = torch.sigmoid(pred)
        inter = (pred_sigmoid * mask).sum(dim=(2, 3))
        total = pred_sigmoid.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice = 1 - (2 * inter + self.smooth) / (total + self.smooth)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice


class GMLFNetLoss(nn.Module):
    """Composite loss with deep supervision for GMLFNet.

    Applies StructureLoss to the main prediction and all side outputs,
    with decreasing weights for side outputs.
    """

    def __init__(self, structure_weight=1.0, side_weights=None):
        super().__init__()
        self.structure_loss = StructureLoss()
        self.structure_weight = structure_weight
        # Weights for side outputs (from deepest to shallowest)
        self.side_weights = side_weights or [0.5, 0.3, 0.2]

    def forward(self, predictions, mask):
        """
        Args:
            predictions: tuple of (main_pred, [side_pred1, side_pred2, ...])
                        All are logits (B, 1, H, W)
            mask: ground truth (B, 1, H, W)
        """
        main_pred, side_preds = predictions

        # Resize mask to match prediction size if needed
        if main_pred.shape[2:] != mask.shape[2:]:
            mask_resized = F.interpolate(
                mask, size=main_pred.shape[2:],
                mode="bilinear", align_corners=False
            )
        else:
            mask_resized = mask

        # Main loss
        total_loss = self.structure_weight * self.structure_loss(main_pred, mask_resized)

        # Side output losses
        for i, side_pred in enumerate(side_preds):
            if i < len(self.side_weights):
                weight = self.side_weights[i]
            else:
                weight = 0.1

            if side_pred.shape[2:] != mask_resized.shape[2:]:
                side_mask = F.interpolate(
                    mask, size=side_pred.shape[2:],
                    mode="bilinear", align_corners=False
                )
            else:
                side_mask = mask_resized

            total_loss += weight * self.structure_loss(side_pred, side_mask)

        return total_loss

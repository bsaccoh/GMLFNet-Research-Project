"""Segmentation evaluation metrics for polyp segmentation.

Standard metrics used in polyp segmentation benchmarks:
- Dice coefficient (F1 score)
- IoU (Jaccard index)
- Precision
- Recall (Sensitivity)
- F-measure (weighted, beta=0.3)
- MAE (Mean Absolute Error)
- S-measure (Structure measure)
- E-measure (Enhanced alignment measure)
"""

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates predictions and computes segmentation metrics.

    All metrics are computed per-image, then averaged.
    Input predictions should be probabilities (after sigmoid), range [0, 1].
    Ground truth masks should be binary {0, 1}.
    """

    def __init__(self, threshold=0.5, beta_sq=0.3):
        self.threshold = threshold
        self.beta_sq = beta_sq  # For F-measure
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.mae_scores = []
        self.smeasure_scores = []
        self.emeasure_scores = []

    @torch.no_grad()
    def update(self, pred, mask):
        """Accumulate one batch of predictions.

        Args:
            pred: predicted probabilities (B, 1, H, W), range [0, 1]
            mask: ground truth binary masks (B, 1, H, W)
        """
        pred = pred.cpu().numpy()
        mask = mask.cpu().numpy()

        for i in range(pred.shape[0]):
            p = pred[i, 0]  # (H, W) probability map
            m = mask[i, 0]  # (H, W) binary mask

            # Binarize prediction
            p_bin = (p >= self.threshold).astype(np.float32)
            m_bin = m.astype(np.float32)

            # Basic counts
            tp = (p_bin * m_bin).sum()
            fp = (p_bin * (1 - m_bin)).sum()
            fn = ((1 - p_bin) * m_bin).sum()

            # Dice
            dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
            self.dice_scores.append(dice)

            # IoU
            iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
            self.iou_scores.append(iou)

            # Precision
            precision = (tp + 1e-8) / (tp + fp + 1e-8)
            self.precision_scores.append(precision)

            # Recall
            recall = (tp + 1e-8) / (tp + fn + 1e-8)
            self.recall_scores.append(recall)

            # MAE
            mae = np.abs(p - m_bin).mean()
            self.mae_scores.append(mae)

            # S-measure
            sm = self._compute_smeasure(p, m_bin)
            self.smeasure_scores.append(sm)

            # E-measure
            em = self._compute_emeasure(p_bin, m_bin)
            self.emeasure_scores.append(em)

    def compute(self):
        """Return all metrics as a dictionary."""
        n = len(self.dice_scores)
        if n == 0:
            return {
                "dice": 0.0, "iou": 0.0, "precision": 0.0,
                "recall": 0.0, "fmeasure": 0.0, "mae": 1.0,
                "smeasure": 0.0, "emeasure": 0.0,
            }

        precision = np.mean(self.precision_scores)
        recall = np.mean(self.recall_scores)
        fmeasure = ((1 + self.beta_sq) * precision * recall + 1e-8) / \
                   (self.beta_sq * precision + recall + 1e-8)

        return {
            "dice": float(np.mean(self.dice_scores)),
            "iou": float(np.mean(self.iou_scores)),
            "precision": float(precision),
            "recall": float(recall),
            "fmeasure": float(fmeasure),
            "mae": float(np.mean(self.mae_scores)),
            "smeasure": float(np.mean(self.smeasure_scores)),
            "emeasure": float(np.mean(self.emeasure_scores)),
        }

    def _compute_smeasure(self, pred, mask, alpha=0.5):
        """Structure measure (Fan et al., 2017).

        Evaluates structural similarity between prediction and ground truth.
        """
        y = mask.mean()

        if y == 0:
            # No foreground: measure background quality
            score = 1.0 - pred.mean()
        elif y == 1:
            # All foreground: measure foreground quality
            score = pred.mean()
        else:
            # Combined object + region structural similarity
            so = self._s_object(pred, mask)
            sr = self._s_region(pred, mask)
            score = alpha * so + (1 - alpha) * sr

        return max(0.0, score)

    def _s_object(self, pred, mask):
        """Object-level structural similarity."""
        # Foreground
        fg_pred = pred * mask
        fg_mask = mask
        o_fg = self._object_score(fg_pred, fg_mask)

        # Background
        bg_pred = (1 - pred) * (1 - mask)
        bg_mask = 1 - mask
        o_bg = self._object_score(bg_pred, bg_mask)

        u = mask.mean()
        return u * o_fg + (1 - u) * o_bg

    def _object_score(self, pred, mask):
        """Compute object score."""
        x = pred[mask > 0.5]
        if len(x) == 0:
            return 0.0
        mu = x.mean()
        std = x.std() + 1e-8
        return 2 * mu / (mu ** 2 + 1 + std + 1e-8)

    def _s_region(self, pred, mask):
        """Region-level structural similarity."""
        h, w = mask.shape
        cx, cy = h // 2, w // 2

        # Split into 4 quadrants
        score = 0.0
        for si, sj in [(slice(0, cx), slice(0, cy)),
                        (slice(0, cx), slice(cy, w)),
                        (slice(cx, h), slice(0, cy)),
                        (slice(cx, h), slice(cy, w))]:
            p_region = pred[si, sj]
            m_region = mask[si, sj]
            weight = m_region.size / (h * w)
            score += weight * self._ssim_like(p_region, m_region)

        return score

    def _ssim_like(self, pred, mask):
        """Simplified SSIM-like measure for a region."""
        mu_p = pred.mean()
        mu_m = mask.mean()
        sigma_p = pred.std() + 1e-8
        sigma_m = mask.std() + 1e-8
        sigma_pm = ((pred - mu_p) * (mask - mu_m)).mean()

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        luminance = (2 * mu_p * mu_m + c1) / (mu_p ** 2 + mu_m ** 2 + c1)
        contrast = (2 * sigma_p * sigma_m + c2) / (sigma_p ** 2 + sigma_m ** 2 + c2)
        structure = (sigma_pm + c2 / 2) / (sigma_p * sigma_m + c2 / 2)

        return luminance * contrast * structure

    def _compute_emeasure(self, pred_bin, mask):
        """Enhanced alignment measure (Fan et al., 2018)."""
        if mask.sum() == 0 and pred_bin.sum() == 0:
            return 1.0
        if mask.sum() == 0 or pred_bin.sum() == 0:
            return 0.0

        # Alignment matrix
        mu_pred = pred_bin.mean()
        mu_mask = mask.mean()

        align_pred = pred_bin - mu_pred
        align_mask = mask - mu_mask

        align_matrix = 2 * (align_pred * align_mask) / \
                       (align_pred ** 2 + align_mask ** 2 + 1e-8)

        enhanced = ((align_matrix + 1) ** 2) / 4
        return enhanced.mean()

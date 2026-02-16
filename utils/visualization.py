"""Visualization utilities for polyp segmentation results."""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_qualitative_results(images, masks, predictions, save_path=None, n_samples=4):
    """Plot side-by-side: input | ground truth | prediction | overlay.

    Args:
        images: (N, 3, H, W) tensor (normalized)
        masks: (N, 1, H, W) tensor (binary)
        predictions: (N, 1, H, W) tensor (probabilities after sigmoid)
        save_path: path to save figure (optional)
        n_samples: number of samples to display
    """
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images.cpu() * std + mean
    images = images.clamp(0, 1)

    masks = masks.cpu()
    predictions = predictions.cpu()
    n = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Input", "Ground Truth", "Prediction", "Overlay"]

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i, 0].numpy()
        pred = predictions[i, 0].numpy()
        pred_bin = (pred >= 0.5).astype(np.float32)

        # Overlay: red for prediction, green for ground truth
        overlay = img.copy()
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
        overlay[pred_bin > 0.5] = overlay[pred_bin > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 3].imshow(overlay)

        for j, title in enumerate(titles):
            axes[i, j].set_title(title if i == 0 else "")
            axes[i, j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_adaptation_curve(metrics_per_step, center_name, save_path=None):
    """Plot Dice improvement over inner-loop adaptation steps.

    Args:
        metrics_per_step: list of dicts, one per adaptation step
        center_name: name of the test center
        save_path: path to save figure
    """
    steps = list(range(len(metrics_per_step)))
    dice_scores = [m["dice"] for m in metrics_per_step]
    iou_scores = [m["iou"] for m in metrics_per_step]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(steps, dice_scores, "b-o", label="Dice", linewidth=2)
    ax.plot(steps, iou_scores, "r-s", label="IoU", linewidth=2)
    ax.set_xlabel("Adaptation Steps")
    ax.set_ylabel("Score")
    ax.set_title(f"Fast Adaptation on {center_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cross_center_comparison(results_dict, metric="dice", save_path=None):
    """Bar chart comparing methods across test centers.

    Args:
        results_dict: dict of method_name -> {center_name: metrics_dict}
        metric: which metric to plot
        save_path: path to save figure
    """
    methods = list(results_dict.keys())
    centers = list(next(iter(results_dict.values())).keys())

    x = np.arange(len(centers))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, method in enumerate(methods):
        values = [results_dict[method][c][metric] for c in centers]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(centers, rotation=15, ha="right")
    ax.legend()
    ax.set_title(f"{metric.capitalize()} Comparison Across Centers")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

"""Entry point: Evaluate a trained model on all test centers.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/best_model.pth
    python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/best_model.pth --mode few_shot
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import learn2learn as l2l

from data.datasets import build_center_datasets
from data.augmentations import get_test_transforms
from models.gmlf_net import build_model
from trainers.evaluator import Evaluator
from utils.misc import load_config, set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="GMLFNet Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot"],
                        help="Evaluation mode")
    parser.add_argument("--output", type=str, default="results/eval_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = get_device()

    # Build test datasets (all 5 centers)
    test_transform = get_test_transforms(cfg.data.image_size)
    all_centers = cfg.data.train_centers + cfg.data.test_centers
    test_datasets = build_center_datasets(
        root=cfg.data.root,
        centers=all_centers,
        transform=test_transform,
        image_size=cfg.data.image_size,
    )

    # Build and load model
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Wrap in MAML for few-shot evaluation
    if args.mode == "few_shot":
        maml_model = l2l.algorithms.MAML(
            model, lr=cfg.meta.inner_lr,
            first_order=cfg.meta.first_order,
        )
        maml_model.to(device)
        eval_model = maml_model
    else:
        eval_model = model

    # Evaluate
    evaluator = Evaluator(test_datasets, cfg, device)
    results = evaluator.full_evaluation(eval_model, mode=args.mode)

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.mode})")
    print(f"{'='*60}")
    print(f"{'Center':<25} {'Dice':>8} {'IoU':>8} {'F-m':>8} {'MAE':>8} {'Sm':>8}")
    print("-" * 60)

    for center, metrics in results.items():
        print(f"{center:<25} "
              f"{metrics['dice']:>8.4f} "
              f"{metrics['iou']:>8.4f} "
              f"{metrics['fmeasure']:>8.4f} "
              f"{metrics['mae']:>8.4f} "
              f"{metrics['smeasure']:>8.4f}")

    # Mean across all centers
    mean_metrics = {}
    for key in results[list(results.keys())[0]]:
        mean_metrics[key] = sum(r[key] for r in results.values()) / len(results)

    print("-" * 60)
    print(f"{'Mean':<25} "
          f"{mean_metrics['dice']:>8.4f} "
          f"{mean_metrics['iou']:>8.4f} "
          f"{mean_metrics['fmeasure']:>8.4f} "
          f"{mean_metrics['mae']:>8.4f} "
          f"{mean_metrics['smeasure']:>8.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

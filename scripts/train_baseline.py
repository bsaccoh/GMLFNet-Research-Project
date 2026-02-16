"""Entry point: Standard supervised training baseline.

Usage:
    python scripts/train_baseline.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data.datasets import build_center_datasets
from data.augmentations import get_train_transforms, get_test_transforms
from models.gmlf_net import build_model
from trainers.baseline_trainer import BaselineTrainer
from trainers.evaluator import Evaluator
from utils.misc import load_config, set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description="GMLFNet Baseline Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = get_device()

    # Build datasets
    train_transform = get_train_transforms(cfg.data.image_size)
    train_datasets = build_center_datasets(
        root=cfg.data.root,
        centers=cfg.data.train_centers,
        transform=train_transform,
        image_size=cfg.data.image_size,
    )

    test_transform = get_test_transforms(cfg.data.image_size)
    all_centers = cfg.data.train_centers + cfg.data.test_centers
    test_datasets = build_center_datasets(
        root=cfg.data.root,
        centers=all_centers,
        transform=test_transform,
        image_size=cfg.data.image_size,
    )

    # Build model (without FAW for fair comparison, or with FAW)
    model = build_model(cfg)
    model.print_param_summary()

    # Trainer and evaluator
    trainer = BaselineTrainer(model, train_datasets, cfg, device)
    evaluator = Evaluator(test_datasets, cfg, device)

    print(f"\nStarting baseline training for {cfg.training.epochs} epochs")
    trainer.train(evaluator=evaluator)


if __name__ == "__main__":
    main()

"""Entry point: Meta-learning training for GMLFNet.

Usage:
    python scripts/train_meta.py --config configs/default.yaml
    python scripts/train_meta.py --config configs/default.yaml --resume runs/checkpoint_epoch50.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data.datasets import build_center_datasets
from data.augmentations import get_train_transforms, get_test_transforms
from data.meta_sampler import CenterEpisodicSampler
from models.gmlf_net import build_model
from trainers.meta_trainer import MAMLMetaTrainer
from trainers.evaluator import Evaluator
from utils.misc import load_config, set_seed, get_device
from utils.logging_utils import Logger


def main():
    parser = argparse.ArgumentParser(description="GMLFNet Meta-Learning Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.resume:
        cfg.training.resume = args.resume

    # Setup
    set_seed(cfg.training.seed)
    device = get_device()

    # Build training datasets (one per center)
    train_transform = get_train_transforms(cfg.data.image_size)
    train_datasets = build_center_datasets(
        root=cfg.data.root,
        centers=cfg.data.train_centers,
        transform=train_transform,
        image_size=cfg.data.image_size,
    )
    print(f"Training centers: {list(train_datasets.keys())}")
    for name, ds in train_datasets.items():
        print(f"  {name}: {len(ds)} images")

    # Build test datasets
    test_transform = get_test_transforms(cfg.data.image_size)
    all_centers = cfg.data.train_centers + cfg.data.test_centers
    test_datasets = build_center_datasets(
        root=cfg.data.root,
        centers=all_centers,
        transform=test_transform,
        image_size=cfg.data.image_size,
    )
    print(f"Test centers: {list(test_datasets.keys())}")

    # Build model
    model = build_model(cfg)
    model.print_param_summary()

    # Build episodic sampler
    sampler = CenterEpisodicSampler(
        center_datasets=train_datasets,
        support_size=cfg.meta.support_size,
        query_size=cfg.meta.query_size,
        device=device,
    )

    # Build trainer and evaluator
    trainer = MAMLMetaTrainer(model, sampler, cfg, device)
    evaluator = Evaluator(test_datasets, cfg, device)

    # Logger
    logger = Logger(
        backend=cfg.logging.backend,
        log_dir=cfg.logging.log_dir,
        project=getattr(cfg.logging, "wandb_project", "GMLFNet"),
        config=cfg.to_dict() if hasattr(cfg, "to_dict") else None,
    )

    # Train
    print(f"\nStarting meta-learning training for {cfg.training.epochs} epochs")
    print(f"  Inner LR: {cfg.meta.inner_lr}, Inner steps: {cfg.meta.inner_steps}")
    print(f"  Outer LR: {cfg.meta.outer_lr}")
    print(f"  First order (FOMAML): {cfg.meta.first_order}")
    print(f"  Support/Query size: {cfg.meta.support_size}/{cfg.meta.query_size}")
    print()

    trainer.train(evaluator=evaluator, logger=logger)
    logger.close()


if __name__ == "__main__":
    main()

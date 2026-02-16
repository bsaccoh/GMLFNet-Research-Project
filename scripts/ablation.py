"""Entry point: Run ablation experiments.

Usage:
    python scripts/ablation.py --config configs/default.yaml --ablation inner_steps
    python scripts/ablation.py --config configs/default.yaml --ablation all
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from data.datasets import build_center_datasets
from data.augmentations import get_train_transforms, get_test_transforms
from data.meta_sampler import CenterEpisodicSampler
from models.gmlf_net import GMLFNet, build_model
from trainers.meta_trainer import MAMLMetaTrainer
from trainers.baseline_trainer import BaselineTrainer
from trainers.evaluator import Evaluator
from utils.misc import load_config, set_seed, get_device


ABLATION_CONFIGS = {
    "inner_steps": {
        "description": "Effect of number of inner-loop adaptation steps",
        "param": "meta.inner_steps",
        "values": [1, 3, 5, 10],
    },
    "faw_hidden_dim": {
        "description": "Effect of FAW hidden dimension",
        "param": "model.faw_hidden_dim",
        "values": [32, 64, 128],
    },
    "support_size": {
        "description": "Effect of support set size",
        "param": "meta.support_size",
        "values": [4, 8, 16, 32],
    },
    "backbone": {
        "description": "Backbone comparison",
        "param": "model.backbone",
        "values": ["res2net50", "pvt_v2_b2"],
    },
    "use_faw": {
        "description": "With vs without FAW module",
        "param": "use_faw",
        "values": [True, False],
    },
}


def set_nested_attr(cfg, key, value):
    """Set a nested attribute on a config object."""
    parts = key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def run_single_experiment(name, cfg, device):
    """Run a single training + evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    set_seed(cfg.training.seed)

    # Build datasets
    train_transform = get_train_transforms(cfg.data.image_size)
    train_datasets = build_center_datasets(
        cfg.data.root, cfg.data.train_centers,
        transform=train_transform, image_size=cfg.data.image_size,
    )

    test_transform = get_test_transforms(cfg.data.image_size)
    all_centers = cfg.data.train_centers + cfg.data.test_centers
    test_datasets = build_center_datasets(
        cfg.data.root, all_centers,
        transform=test_transform, image_size=cfg.data.image_size,
    )

    # Build model
    use_faw = getattr(cfg, "use_faw", True) if hasattr(cfg, "use_faw") else True
    model = GMLFNet(
        backbone_name=cfg.model.backbone,
        decoder_channel=cfg.model.decoder_channels[-1],
        faw_hidden_dim=cfg.model.faw_hidden_dim,
        faw_num_layers=cfg.model.faw_num_layers,
        pretrained=True,
        use_faw=use_faw,
    )

    # Build sampler and trainer
    sampler = CenterEpisodicSampler(
        train_datasets,
        support_size=cfg.meta.support_size,
        query_size=cfg.meta.query_size,
        device=device,
    )
    trainer = MAMLMetaTrainer(model, sampler, cfg, device)
    evaluator = Evaluator(test_datasets, cfg, device)

    # Train (shorter for ablation)
    trainer.train(evaluator=evaluator)

    # Final evaluation
    results = evaluator.full_evaluation(trainer.maml)
    return results


def run_ablation(ablation_name, base_cfg, device):
    """Run all variants of a single ablation study."""
    abl = ABLATION_CONFIGS[ablation_name]
    all_results = {}

    print(f"\nAblation: {abl['description']}")

    for value in abl["values"]:
        cfg = copy.deepcopy(base_cfg)

        if abl["param"] == "use_faw":
            cfg.use_faw = value
            exp_name = f"{ablation_name}_faw={value}"
        else:
            set_nested_attr(cfg, abl["param"], value)
            exp_name = f"{ablation_name}_{abl['param'].split('.')[-1]}={value}"

        # Shorter training for ablation
        cfg.training.epochs = min(cfg.training.epochs, 50)
        cfg.logging.log_dir = f"./runs/ablation/{exp_name}"

        results = run_single_experiment(exp_name, cfg, device)
        all_results[exp_name] = results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="GMLFNet Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATION_CONFIGS.keys()) + ["all"])
    parser.add_argument("--output", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    all_results = {}

    if args.ablation == "all":
        for abl_name in ABLATION_CONFIGS:
            results = run_ablation(abl_name, cfg, device)
            all_results.update(results)
    else:
        results = run_ablation(args.ablation, cfg, device)
        all_results.update(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for exp_name, center_results in all_results.items():
        serializable[exp_name] = {
            center: metrics for center, metrics in center_results.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll ablation results saved to {output_path}")


if __name__ == "__main__":
    main()

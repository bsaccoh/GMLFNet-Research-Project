"""Standard supervised training baseline (no meta-learning).

Trains on pooled data from all training centers using standard SGD/Adam.
Serves as comparison baseline for meta-learning experiments.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models.losses import GMLFNetLoss
from utils.misc import save_checkpoint


class BaselineTrainer:
    """Standard supervised trainer for GMLFNet (no meta-learning).

    Args:
        model: GMLFNet instance
        train_datasets: dict of center_name -> PolypCenterDataset
        cfg: Config object
        device: torch device
    """

    def __init__(self, model, train_datasets, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # Pool all training datasets
        all_datasets = list(train_datasets.values())
        self.train_dataset = ConcatDataset(all_datasets)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.meta.support_size,  # reuse batch size
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.meta.outer_lr,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=1e-6,
        )

        # Loss
        self.loss_fn = GMLFNetLoss()

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.loss_fn(predictions, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip,
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def train(self, evaluator=None):
        """Full training loop."""
        best_dice = 0.0

        for epoch in range(self.cfg.training.epochs):
            t0 = time.time()
            avg_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            print(f"Epoch {epoch}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

            if evaluator and (epoch + 1) % self.cfg.logging.save_interval == 0:
                results = evaluator.full_evaluation(self.model)
                mean_dice = sum(
                    r["dice"] for r in results.values()
                ) / len(results)

                print(f"  Eval mean_dice={mean_dice:.4f}")
                for center, metrics in results.items():
                    print(f"    {center}: dice={metrics['dice']:.4f}")

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        {"mean_dice": mean_dice},
                        Path(self.cfg.logging.log_dir) / "best_baseline.pth",
                    )

        print(f"\nBaseline training complete. Best mean Dice: {best_dice:.4f}")

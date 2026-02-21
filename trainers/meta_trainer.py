"""MAML-based meta-training for multi-centre polyp segmentation.

Training protocol:
1. Each episode samples one task per training center
2. Inner loop: clone model, adapt on support set (K gradient steps)
   - Only FAW parameters are adapted (selective adaptation)
3. Outer loop: compute query loss on adapted model, backprop to all params
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import learn2learn as l2l
from tqdm import tqdm

from data.meta_sampler import CenterEpisodicSampler
from models.losses import GMLFNetLoss
from utils.misc import save_checkpoint


class MAMLMetaTrainer:
    """MAML meta-trainer for GMLFNet.

    Args:
        model: GMLFNet instance
        sampler: CenterEpisodicSampler for episodic training data
        cfg: Config object
        device: torch device
    """

    def __init__(self, model, sampler, cfg, device):
        self.cfg = cfg
        self.device = device
        self.sampler = sampler

        # Wrap model with learn2learn MAML
        self.maml = l2l.algorithms.MAML(
            model,
            lr=cfg.meta.inner_lr,
            first_order=cfg.meta.first_order,
            allow_unused=True,
            allow_nograd=True,
        )
        self.maml.to(device)

        # Outer-loop optimizer
        self.outer_optimizer = torch.optim.Adam(
            self.maml.parameters(),
            lr=cfg.meta.outer_lr,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.outer_optimizer,
            T_max=cfg.training.epochs,
            eta_min=1e-6,
        )

        # Loss function
        self.loss_fn = GMLFNetLoss()

        # Inner loop config
        self.inner_steps = cfg.meta.inner_steps
        self.selective_adaptation = True  # Only adapt FAW in inner loop

    def meta_train_step(self):
        """One meta-training iteration (one episode).

        Returns:
            meta_loss: scalar loss value
            task_losses: dict of per-center query losses
        """
        self.outer_optimizer.zero_grad()
        meta_loss = 0.0
        task_losses = {}

        # Sample episode: one task per training center
        episode = self.sampler.sample_episode()

        for task in episode:
            learner = self.maml.clone()

            # === Inner Loop (Adaptation) ===
            if self.selective_adaptation:
                # Freeze everything except FAW
                for name, param in learner.named_parameters():
                    if "faw" not in name:
                        param.requires_grad_(False)
                    else:
                        param.requires_grad_(True)

            for step in range(self.inner_steps):
                support_pred = learner(task.support_images)
                support_loss = self.loss_fn(support_pred, task.support_masks)
                learner.adapt(support_loss)

            # === Outer Loop (Meta-Objective) ===
            # Re-enable all gradients for outer loop
            if self.selective_adaptation:
                for param in learner.parameters():
                    param.requires_grad_(True)

            query_pred = learner(task.query_images)
            query_loss = self.loss_fn(query_pred, task.query_masks)
            meta_loss += query_loss
            task_losses[task.center_name] = query_loss.item()

        meta_loss /= len(episode)
        meta_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.maml.parameters(),
            self.cfg.training.grad_clip,
        )
        self.outer_optimizer.step()

        return meta_loss.item(), task_losses

    def train_epoch(self, epoch):
        """Train for one epoch.

        Args:
            epoch: current epoch number

        Returns:
            avg_loss: average meta-loss for the epoch
        """
        self.maml.train()
        steps_per_epoch = len(self.sampler)
        total_loss = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            loss, task_losses = self.meta_train_step()
            total_loss += loss

            # Update progress bar
            task_str = " | ".join(
                f"{k[:8]}:{v:.4f}" for k, v in task_losses.items()
            )
            pbar.set_postfix_str(f"loss={loss:.4f} | {task_str}")

        self.scheduler.step()
        avg_loss = total_loss / steps_per_epoch
        return avg_loss

    def train(self, evaluator=None, logger=None):
        """Full training loop.

        Args:
            evaluator: Evaluator instance for validation
            logger: logging utility
        """
        best_dice = 0.0
        start_epoch = 0

        # Resume from checkpoint if specified
        if self.cfg.training.resume:
            checkpoint_path = Path(self.cfg.training.resume)
            if checkpoint_path.exists():
                start_epoch, metrics = load_checkpoint(
                    self.maml.module,
                    optimizer=self.outer_optimizer,
                    path=checkpoint_path
                )
                best_dice = metrics.get("mean_dice", 0.0)
                start_epoch += 1  # Resume from next epoch
                print(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")

        for epoch in range(start_epoch, self.cfg.training.epochs):
            t0 = time.time()
            avg_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s, "
                  f"lr={self.scheduler.get_last_lr()[0]:.6f}")

            # Evaluate periodically
            if evaluator and (epoch + 1) % self.cfg.logging.save_interval == 0:
                results = evaluator.full_evaluation(self.maml)
                mean_dice = sum(
                    r["dice"] for r in results.values()
                ) / len(results)

                print(f"  Eval mean_dice={mean_dice:.4f}")
                for center, metrics in results.items():
                    print(f"    {center}: dice={metrics['dice']:.4f}, "
                          f"iou={metrics['iou']:.4f}")

                # Save best checkpoint
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    save_checkpoint(
                        self.maml.module,
                        self.outer_optimizer,
                        epoch,
                        {"mean_dice": mean_dice, **results},
                        Path(self.cfg.logging.log_dir) / "best_model.pth",
                    )

            # Save periodic checkpoint (for Colab session recovery)
            if (epoch + 1) % self.cfg.logging.save_interval == 0:
                save_checkpoint(
                    self.maml.module,
                    self.outer_optimizer,
                    epoch,
                    {"avg_loss": avg_loss},
                    Path(self.cfg.logging.log_dir) / f"checkpoint_epoch{epoch}.pth",
                )

        print(f"\nTraining complete. Best mean Dice: {best_dice:.4f}")

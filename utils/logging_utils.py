"""Logging utilities for TensorBoard and Weights & Biases."""

from pathlib import Path


class Logger:
    """Unified logging interface for TensorBoard and W&B.

    Args:
        backend: "tensorboard" or "wandb"
        log_dir: directory for TensorBoard logs
        project: W&B project name
        config: config dict for W&B init
    """

    def __init__(self, backend="tensorboard", log_dir="./runs",
                 project="GMLFNet", config=None):
        self.backend = backend

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        elif backend == "wandb":
            import wandb
            wandb.init(project=project, config=config)
            self.writer = wandb
        else:
            self.writer = None

    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        if self.backend == "tensorboard" and self.writer:
            self.writer.add_scalar(tag, value, step)
        elif self.backend == "wandb" and self.writer:
            self.writer.log({tag: value}, step=step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under a group."""
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(f"{main_tag}/{tag}", value, step)

    def log_image(self, tag, image, step):
        """Log an image (numpy array HxWx3 or tensor CxHxW)."""
        if self.backend == "tensorboard" and self.writer:
            if hasattr(image, "numpy"):
                image = image.numpy()
            self.writer.add_image(tag, image, step, dataformats="HWC")
        elif self.backend == "wandb" and self.writer:
            import wandb
            self.writer.log({tag: wandb.Image(image)}, step=step)

    def close(self):
        """Close the logger."""
        if self.backend == "tensorboard" and self.writer:
            self.writer.close()
        elif self.backend == "wandb" and self.writer:
            self.writer.finish()

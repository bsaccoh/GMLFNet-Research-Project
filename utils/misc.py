import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


class Config:
    """Simple nested config object from a dictionary."""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            elif isinstance(v, list):
                setattr(self, k, [Config(i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_config(config_path="configs/default.yaml"):
    """Load YAML config file into a Config object."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    # Allow environment variable override for data root
    data_root_env = os.environ.get("GMLFNET_DATA_ROOT")
    if data_root_env and "data" in cfg_dict:
        cfg_dict["data"]["root"] = data_root_env
        
    return Config(cfg_dict)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint.

    Args:
        model: the model (or MAML wrapper)
        optimizer: optimizer state
        epoch: current epoch
        metrics: dict of evaluation metrics
        path: save path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer=None, path=None):
    """Load training checkpoint.

    Args:
        model: the model to load weights into
        optimizer: optimizer to load state into (optional)
        path: checkpoint path

    Returns:
        epoch, metrics from the checkpoint
    """
    if path is None or not Path(path).exists():
        print("No checkpoint found, starting from scratch.")
        return 0, {}

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error loading checkpoint with weights_only=True: {e}. Falling back to weights_only=False.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics

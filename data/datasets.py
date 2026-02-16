import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import get_train_transforms, get_test_transforms


class PolypCenterDataset(Dataset):
    """Dataset for a single polyp segmentation center.

    Expects folder structure:
        root/center_name/images/*.png (or .jpg)
        root/center_name/masks/*.png (or .jpg)

    Image and mask files must have matching filenames.
    """

    def __init__(self, root, center_name, transform=None, image_size=352):
        self.root = Path(root)
        self.center_name = center_name
        self.image_size = image_size
        self.transform = transform

        self.image_dir = self.root / center_name / "images"
        self.mask_dir = self.root / center_name / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        # Collect image filenames (support common formats)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in valid_exts
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        # Verify matching masks exist
        self.mask_files = []
        for img_path in self.image_files:
            mask_path = self._find_mask(img_path)
            if mask_path is None:
                raise FileNotFoundError(
                    f"No matching mask for {img_path.name} in {self.mask_dir}"
                )
            self.mask_files.append(mask_path)

    def _find_mask(self, img_path):
        """Find the corresponding mask file (may differ in extension)."""
        stem = img_path.stem
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
            mask_path = self.mask_dir / f"{stem}{ext}"
            if mask_path.exists():
                return mask_path
        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Read image (BGR -> RGB) and mask (grayscale)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Binarize mask (threshold at 128)
        mask = (mask > 128).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]  # (C, H, W) float tensor
            mask = transformed["mask"]    # (H, W) float tensor
        else:
            # Default: resize and convert to tensor
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Ensure mask has channel dimension: (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "center": self.center_name,
            "filename": img_path.name,
        }


def build_center_datasets(root, centers, transform=None, image_size=352):
    """Build a dictionary of datasets, one per center.

    Args:
        root: path to dataset root
        centers: list of center names
        transform: albumentations transform
        image_size: target image size

    Returns:
        dict mapping center_name -> PolypCenterDataset
    """
    datasets = {}
    for center in centers:
        datasets[center] = PolypCenterDataset(
            root=root,
            center_name=center,
            transform=transform,
            image_size=image_size,
        )
    return datasets

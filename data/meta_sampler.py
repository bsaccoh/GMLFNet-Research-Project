import random
from collections import namedtuple
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .datasets import PolypCenterDataset


Task = namedtuple("Task", [
    "support_images",
    "support_masks",
    "query_images",
    "query_masks",
    "center_name",
])


class CenterEpisodicSampler:
    """Creates meta-learning episodes where each task = one center.

    Each episode produces one Task per training center. Each Task contains
    a support set (for inner-loop adaptation) and a query set (for
    outer-loop loss computation).

    Args:
        center_datasets: dict mapping center_name -> PolypCenterDataset
        support_size: number of images in the support set per task
        query_size: number of images in the query set per task
        device: target device for tensors
    """

    def __init__(
        self,
        center_datasets: Dict[str, PolypCenterDataset],
        support_size: int = 16,
        query_size: int = 16,
        device: torch.device = None,
    ):
        self.center_datasets = center_datasets
        self.support_size = support_size
        self.query_size = query_size
        self.device = device or torch.device("cpu")
        self.center_names = list(center_datasets.keys())

        # Validate that each center has enough images
        total_needed = support_size + query_size
        for name, ds in center_datasets.items():
            if len(ds) < total_needed:
                print(
                    f"Warning: {name} has {len(ds)} images but "
                    f"{total_needed} needed per episode. "
                    f"Sampling with replacement."
                )

    def sample_episode(self) -> List[Task]:
        """Sample one episode: one task per training center.

        Returns:
            List of Task namedtuples, one per center.
        """
        tasks = []
        for center_name in self.center_names:
            task = self._sample_task(center_name)
            tasks.append(task)
        return tasks

    def _sample_task(self, center_name: str) -> Task:
        """Sample support and query sets from a single center."""
        dataset = self.center_datasets[center_name]
        total_needed = self.support_size + self.query_size
        n = len(dataset)

        # Sample indices (with replacement if dataset is too small)
        if n >= total_needed:
            indices = random.sample(range(n), total_needed)
        else:
            indices = random.choices(range(n), k=total_needed)

        support_indices = indices[:self.support_size]
        query_indices = indices[self.support_size:]

        support_images, support_masks = self._collate(dataset, support_indices)
        query_images, query_masks = self._collate(dataset, query_indices)

        return Task(
            support_images=support_images.to(self.device),
            support_masks=support_masks.to(self.device),
            query_images=query_images.to(self.device),
            query_masks=query_masks.to(self.device),
            center_name=center_name,
        )

    def _collate(self, dataset, indices):
        """Collate samples into batched tensors."""
        images = []
        masks = []
        for idx in indices:
            sample = dataset[idx]
            images.append(sample["image"])
            masks.append(sample["mask"])
        return torch.stack(images), torch.stack(masks)

    def __len__(self):
        """Approximate number of episodes per epoch.

        Defined as max dataset size / (support + query), so that
        the largest center sees most of its data each epoch.
        """
        max_size = max(len(ds) for ds in self.center_datasets.values())
        return max(1, max_size // (self.support_size + self.query_size))

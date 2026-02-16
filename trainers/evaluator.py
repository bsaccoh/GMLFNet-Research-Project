"""Evaluation for GMLFNet.

Two evaluation modes:
1. Zero-shot: direct inference on unseen centers (tests generalization)
2. Few-shot: K inner-loop adaptation steps on support set, then evaluate
   (tests fast adaptation capability)
"""

import torch
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import DataLoader

from utils.metrics import SegmentationMetrics
from models.losses import GMLFNetLoss


class Evaluator:
    """Evaluator for polyp segmentation models.

    Args:
        test_datasets: dict of center_name -> PolypCenterDataset
        cfg: Config object
        device: torch device
    """

    def __init__(self, test_datasets, cfg, device):
        self.test_datasets = test_datasets
        self.cfg = cfg
        self.device = device
        self.loss_fn = GMLFNetLoss()

    @torch.no_grad()
    def evaluate_zero_shot(self, model, center_name):
        """Evaluate model on a test center without adaptation.

        Args:
            model: the model (GMLFNet or MAML wrapper)
            center_name: name of the test center

        Returns:
            dict of metric values
        """
        # Handle MAML wrapper
        if isinstance(model, l2l.algorithms.MAML):
            eval_model = model.module
        else:
            eval_model = model

        eval_model.eval()
        dataset = self.test_datasets[center_name]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        metrics = SegmentationMetrics()

        for batch in loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)

            main_pred, _ = eval_model(image)
            pred = torch.sigmoid(main_pred)

            # Resize prediction to match mask if needed
            if pred.shape[2:] != mask.shape[2:]:
                pred = F.interpolate(
                    pred, size=mask.shape[2:],
                    mode="bilinear", align_corners=False,
                )

            metrics.update(pred, mask)

        return metrics.compute()

    def evaluate_few_shot(self, maml_model, center_name,
                          k_support=5, adaptation_steps=5):
        """Evaluate with few-shot adaptation.

        Splits test data into support (for adaptation) and query (for eval).

        Args:
            maml_model: MAML-wrapped model
            center_name: test center name
            k_support: number of support images for adaptation
            adaptation_steps: number of inner-loop steps

        Returns:
            dict of metric values
        """
        dataset = self.test_datasets[center_name]
        n = len(dataset)

        if n <= k_support:
            print(f"Warning: {center_name} has only {n} images, "
                  f"cannot split into support/query with k={k_support}")
            return self.evaluate_zero_shot(maml_model, center_name)

        # Split into support and query
        indices = list(range(n))
        support_indices = indices[:k_support]
        query_indices = indices[k_support:]

        # Collect support set
        support_images = []
        support_masks = []
        for idx in support_indices:
            sample = dataset[idx]
            support_images.append(sample["image"])
            support_masks.append(sample["mask"])
        support_images = torch.stack(support_images).to(self.device)
        support_masks = torch.stack(support_masks).to(self.device)

        # Clone and adapt
        learner = maml_model.clone()

        # Selective adaptation: only FAW
        for name, param in learner.named_parameters():
            if "faw" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        for step in range(adaptation_steps):
            pred = learner(support_images)
            loss = self.loss_fn(pred, support_masks)
            learner.adapt(loss)

        # Evaluate on query set
        learner.eval()
        metrics = SegmentationMetrics()

        with torch.no_grad():
            for idx in query_indices:
                sample = dataset[idx]
                image = sample["image"].unsqueeze(0).to(self.device)
                mask = sample["mask"].unsqueeze(0).to(self.device)

                main_pred, _ = learner(image)
                pred = torch.sigmoid(main_pred)

                if pred.shape[2:] != mask.shape[2:]:
                    pred = F.interpolate(
                        pred, size=mask.shape[2:],
                        mode="bilinear", align_corners=False,
                    )

                metrics.update(pred, mask)

        return metrics.compute()

    def full_evaluation(self, model, mode="zero_shot"):
        """Evaluate on all test centers.

        Args:
            model: model or MAML wrapper
            mode: "zero_shot" or "few_shot"

        Returns:
            dict of center_name -> metrics dict
        """
        results = {}
        for center_name in self.test_datasets:
            if mode == "few_shot" and isinstance(model, l2l.algorithms.MAML):
                results[center_name] = self.evaluate_few_shot(model, center_name)
            else:
                results[center_name] = self.evaluate_zero_shot(model, center_name)
        return results

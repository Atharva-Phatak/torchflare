"""Implementations of mixup , cutmix."""
from typing import Callable, Tuple

import numpy as np
import torch


def mixup(batch: Tuple[torch.Tensor, torch.Tensor], alpha: float = 1.0) -> Tuple:
    """Function to mixup data.

    Mixup: <https://arxiv.org/abs/1710.09412>

    Args:
        batch : Tuple containing the data and targets.
        alpha : beta distribution a=b parameters.

    Returns:
        The mixed image and targets.
    """
    data, targets = batch

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    indices = torch.randperm(data.shape[0])
    mixed_data = lam * data + (1 - lam) * data[indices, :]
    target_a, target_b = targets, targets[indices]

    targets = (target_a, target_b, lam)

    return mixed_data, targets


def random_bbox(data, lam):
    """Function to crop random bboxes.

    Args:
        data: The input data.
        lam: The beta distribution value.

    Returns:
        Co-ordinates of bbox.
    """
    img_h, img_w = data.shape[2:]
    cx = np.random.uniform(0, img_w)
    cy = np.random.uniform(0, img_h)
    w = img_w * np.sqrt(1 - lam)
    h = img_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, img_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, img_h)))

    return x0, x1, y0, y1


def cutmix(batch: Tuple[torch.Tensor, torch.Tensor], alpha: float = 1.0) -> Tuple:
    """Function to perform cutmix.

    Cutmix: <https://arxiv.org/abs/1905.04899>

    Args:
        batch : Tuple containing the data and targets.
        alpha : beta distribution a=b parameters.

    Returns:
        Image and targets.
    """
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    x0, x1, y0, y1 = random_bbox(data, lam)

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    targets = (targets, shuffled_targets, lam)

    return data, targets


class MixCriterion:
    """Class to calculate loss when batch mixers are used."""

    def __init__(self, criterion: Callable):
        """Constructor Class for MixCriterion.

        Args:
            criterion: The criterion to be used.
        """
        self.criterion = criterion

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Method to calculate loss.

        Args:
            preds: The output of network.
            targets: The targets.

        Returns:
            The computed loss.
        """
        if isinstance(targets, (list, tuple)):

            target_a, target_b, lam = targets
            loss = lam * self.criterion(preds, target_a) + (1 - lam) * self.criterion(preds, target_b)
        else:
            loss = self.criterion(preds, targets)
        return loss


class CustomCollate:
    """Class to create custom collate_fn for dataloaders."""

    def __init__(self, mixer: Callable, alpha: float = 1.0):
        """Constructor for CustomCollate class.

        Args:
            mixer: The batch mix function to be used.
            alpha: beta distribution a=b parameters.
        """
        self.alpha = alpha
        self.aug = mixer

    def __call__(self, batch):
        """Call method.

        Args:
            batch : The input batch from dataloader.

        Returns:
            Batch with a mixer applied.
        """
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = self.aug(batch, self.alpha)

        return batch


def get_collate_fn(mixer_name: str, alpha: float) -> Callable:
    """Method to create  collate_fn for dataloader.

    Args:
        mixer_name: The name of the batch_mixer.
        alpha: beta distribution a=b parameters.

    Returns:
        The collate_fn for the respective special augmentation.

    Note:
        aug_name must be one of cutmix , mixup
    """
    fn = cutmix if mixer_name == "cutmix" else mixup
    collate_fn = CustomCollate(alpha=alpha, mixer=fn)
    return collate_fn


__all__ = ["mixup", "cutmix", "CustomCollate", "MixCriterion", "get_collate_fn"]

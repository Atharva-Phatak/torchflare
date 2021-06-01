"""Simple utilities required by experiment."""
from typing import TYPE_CHECKING

import numpy as np
import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class AvgLoss(Callbacks):
    """Class for averaging the loss."""

    def __init__(self):
        super(AvgLoss, self).__init__(order=CallbackOrder.LOSS)
        self.total, self.count = 0, 0
        self.reset()

    def reset(self):
        """Reset the variables."""
        self.total, self.count = 0, 0

    def on_batch_end(self, experiment: "Experiment"):
        """Accumulate values."""
        bs = experiment.dataloaders.get(experiment.stage).batch_size
        self.total += experiment.loss.item() * bs
        self.count += bs

    def on_loader_end(self, experiment: "Experiment"):
        """Method to return computed dictionary."""
        prefix = experiment.get_prefix()
        loss_dict = {prefix + "loss": self.total / self.count}
        self.reset()
        experiment.monitors[experiment.stage] = loss_dict


def _has_intersection(key, event):
    return True if key in event else False


def to_device(value, device):
    """Move tensor, list of tensors, list of list of tensors, dict of tensors, tuple of tensors to target device.

    Args:
        value: Object to be moved to the device.
        device: target device.

    Returns:
        Same structure as value, but all tensors and np.arrays moved to device
    """
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [to_device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: to_device(value[k], device) for k in value.dtype.fields.keys()}
    elif isinstance(value, np.ndarray):
        return torch.tensor(value, device=device)
    return value


def _apply_fn(x):

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x


def numpy_to_torch(value):
    """Converts list of np.arrays , dict of np.arrays , np.arrays to tensor."""
    if isinstance(value, dict):
        return {k: _apply_fn(v) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [_apply_fn(v) for v in value]
    elif isinstance(value, np.ndarray):
        return _apply_fn(value)
    else:
        return None


def to_numpy(x):
    """Convert tensors to numpy array.

    Args:
        x : The input tensor.

    Returns:
        Numpy array.
    """
    return x.detach().cpu().numpy()

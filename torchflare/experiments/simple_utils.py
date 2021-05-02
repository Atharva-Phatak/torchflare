"""Simple utilities required by experiment."""
from typing import List

import numpy as np
import torch


class AvgLoss:
    """Class for averaging the loss."""

    def __init__(self):
        self.total, self.count = 0, 0
        self.loss_dict = None
        self.exp = None
        self.reset()

    def reset(self):
        """Reset the variables."""
        self.total, self.count = 0, 0

    def set_experiment(self, exp):
        """Set experiments."""
        self.exp = exp

    def accumulate(self):
        """Accumulate values."""
        bs = self.exp.train_dl.batch_size if self.exp.is_training else self.exp.valid_dl.batch_size
        self.total += self.exp.loss.item() * bs
        self.count += bs

    @property
    def value(self):
        """Method to return computed dictionary."""
        self.loss_dict = {self.exp.get_prefix() + "loss": self.total / self.count}
        self.reset()
        return self.loss_dict


def wrap_metric_names(metric_list: List):
    """Method to  get the metric names from metric list.

    Args:
        metric_list : A list of metrics.

    Returns:
        list of metric name.s
    """
    return [metric.handle() for metric in metric_list]


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


def to_numpy(x):
    """Convert tensors to numpy array.

    Args:
        x : The input tensor.

    Returns:
        Numpy array.
    """
    return x.detach().cpu().numpy()

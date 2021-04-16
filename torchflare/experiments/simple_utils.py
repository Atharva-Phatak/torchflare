"""Simple utilities required by experiment."""
from typing import List

import numpy as np
import torch


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

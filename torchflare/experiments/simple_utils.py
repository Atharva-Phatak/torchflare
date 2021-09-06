"""Simple utilities required by experiment."""
import numpy as np
import torch

from torchflare.callbacks.callback_decorators import FunctionalCallback


def _pop_item(x):
    if torch.is_tensor(x):
        return x.item()
    return x


def _has_intersection(key, event):
    return True if key in event else False


def to_device(value, device):
    """Move tensor, list of tensors, \
    list of list of tensors, dict of tensors, \
    tuple of tensors to target device.

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


def check_same_keys(d1, d2):
    """Check if dictionaries have same keys or not.

    Args:
        d1: The first dict.
        d2: The second dict.

    Raises:
        ValueError if both keys do not match.
    """
    if d1.keys() == d2.keys():
        return True
    raise ValueError("Keys for both the dictionaries should be the same.")


def check_both_dicts(d1, d2):
    """Method to check if both inputs are dictionaries or not.

    Args:
        d1: The first dict.
        d2: The second dict.
    """
    if isinstance(d1, dict) and isinstance(d2, dict):
        return True
    return False


def get_name(obj):
    """Method to get name of the object.

    Args:
        obj: The input object.
    """
    if isinstance(obj, FunctionalCallback):
        name = obj.__name__
    else:
        name = type(obj).__name__
    return name

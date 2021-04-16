"""Implements get optimizer method."""
import torch


def get_optimizer(optimizer):
    """Method to get optimizer from pytorch."""
    dir_optim = dir(torch.optim)
    opts = [o.lower() for o in dir_optim]

    if isinstance(optimizer, str):

        try:
            str_idx = opts.index(optimizer.lower())
            return getattr(torch.optim, dir_optim[str_idx])
        except ValueError:
            raise ValueError("Invalid optimizer string input - must match pytorch optimizer in torch.optim")

    elif hasattr(optimizer, "step") and hasattr(optimizer, "zero_grad"):

        return optimizer

    else:

        raise ValueError("Invalid optimizer input")

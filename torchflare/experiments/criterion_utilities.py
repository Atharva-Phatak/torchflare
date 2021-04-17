"""Implements get criterion method."""
import torch.nn.functional as F


def get_criterion(criterion):
    """Method to get criterion from nn.functional.

    Args:
        criterion: The criterion either str or callable.

    Returns:
        The Required criterion.

    Raises:
        ValueError: If input is string and criterion is not found in nn.functional value error is raised.
    """
    dir_f = dir(F)
    loss_fns = [d.lower() for d in dir_f]

    if isinstance(criterion, str):
        try:
            idx = loss_fns.index(criterion.lower())
            crit = getattr(F, dir(F)[idx])
            return crit
        except ValueError:
            raise ValueError("Invalid loss string input - must match pytorch function in torch.nn.functional")

    elif callable(criterion):

        return criterion

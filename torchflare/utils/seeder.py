"""Implements function for seeding."""
import os
import random

import numpy as np
import torch


def seed_all(seed: int = 42):
    """Method to seed the experiment.

    Args:
        seed(int): The value for the seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)

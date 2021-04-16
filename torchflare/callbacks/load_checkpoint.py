"""Implements Load checkpoint."""
from abc import ABC

import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class LoadCheckpoint(Callbacks, ABC):
    """Class to load checkpoint."""

    def __init__(self):
        """Constructor method for LoadCheckpoint Class."""
        super(LoadCheckpoint, self).__init__(order=CallbackOrder.INTERNAL)

    def experiment_start(self):
        """Load checkpoint before starting training."""
        checkpoint = torch.load(self.exp.path, map_location=torch.device(self.exp.device))
        self.exp.model.load_state_dict(checkpoint["model_state_dict"])
        self.exp.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.exp.scheduler is not None:
            self.exp.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Successfully loaded checkpoints.")

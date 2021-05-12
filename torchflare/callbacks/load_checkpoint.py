"""Implements Load checkpoint."""
from abc import ABC

import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class LoadCheckpoint(Callbacks, ABC):
    """Class to load checkpoint."""

    def __init__(self, path_to_model: str = None):
        """Constructor method for LoadCheckpoint Class."""
        super(LoadCheckpoint, self).__init__(order=CallbackOrder.INTERNAL)
        self.path = path_to_model

    def on_experiment_start(self):
        """Load checkpoint before starting training."""
        checkpoint = torch.load(self.path, map_location=torch.device(self.exp.device))
        self.exp.model.load_state_dict(checkpoint["model_state_dict"])
        self.exp.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.exp.scheduler_stepper is not None:
            self.exp.scheduler_stepper.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print("Successfully loaded checkpoints.")

"""Implements Load checkpoint."""
from abc import ABC
from typing import TYPE_CHECKING

import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class LoadCheckpoint(Callbacks, ABC):
    """Class to load checkpoint."""

    def __init__(self, path_to_model: str = None):
        """Constructor method for LoadCheckpoint Class."""
        super(LoadCheckpoint, self).__init__(order=CallbackOrder.MODEL_INIT)
        self.path = path_to_model

    def on_experiment_start(self, experiment: "Experiment"):
        """Load checkpoint before starting training."""
        checkpoint = torch.load(self.path, map_location=torch.device(experiment.device))
        experiment.model.load_state_dict(checkpoint["model_state_dict"])
        experiment.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Successfully loaded checkpoints.")

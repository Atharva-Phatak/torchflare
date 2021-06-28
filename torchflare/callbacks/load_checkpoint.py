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

    @staticmethod
    def unpack_ckpt(nn_obj, ckpt):
        """Method to unpack checkpoint.

        Args:
            nn_obj: The nn.Module object.
            ckpt: The corresponding state_dict for the object.
        """
        if isinstance(nn_obj, dict):
            for k, v in nn_obj.items():
                v.load_state_dict(ckpt[k])
        else:
            nn_obj.load_state_dict(ckpt)

    def on_experiment_start(self, experiment: "Experiment"):
        """Load checkpoint before starting training."""
        checkpoint = torch.load(self.path, map_location=torch.device(experiment.device))
        self.unpack_ckpt(nn_obj=experiment.state.model, ckpt=checkpoint["model_state_dict"])
        self.unpack_ckpt(nn_obj=experiment.state.optimizer, ckpt=checkpoint["optimizer_state_dict"])

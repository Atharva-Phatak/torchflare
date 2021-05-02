"""Implements Model Checkpoint Callback."""
from abc import ABC

import numpy as np
import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.extra_utils import init_improvement
from torchflare.callbacks.states import CallbackOrder


class ModelCheckpoint(Callbacks, ABC):
    """Callback for Checkpointing your model."""

    def __init__(self, mode: str, monitor: str = "val_loss"):
        """Constructor for ModelCheckpoint class.

        Args:
            mode: One of {"min", "max"}.
                In min mode, training will stop when the quantity monitored has stopped decreasing
                in "max" mode it will stop when the quantity monitored has stopped increasing.
            monitor: The quantity to be monitored. (Default : val_loss)
                    If you want to monitor other metric just pass in the name of the metric.

        Note:

             ModelCheckpoint will save state_dictionaries for model , optimizer , scheduler
             and the value of epoch with following key values:

            1) 'model_state_dict' : The state dictionary of model
            2) 'optimizer_state_dict'  : The state dictionary of optimizer
            4) 'Epoch' : The epoch at which model was saved.

            Model checkpoint will be saved based on the values of metrics/loss obtained from validation set.
        """
        super(ModelCheckpoint, self).__init__(order=CallbackOrder.INTERNAL)
        self.mode = mode
        self.eps = 1e-7
        if "val_" not in monitor:
            self.monitor = "val_" + monitor
        else:
            self.monitor = monitor

        self.improvement, self.best_val = init_improvement(mode=self.mode, min_delta=self.eps)

    def checkpoint(self, epoch: int):
        """Method to save the state dictionaries of model, optimizer,etc.

        Args:
            epoch : The epoch at which model is saved.
        """
        torch.save(
            {
                "model_state_dict": self.exp.model.state_dict(),
                "optimizer_state_dict": self.exp.optimizer.state_dict(),
                "Epoch": epoch,
            },
            self.exp.path,
        )

    def epoch_end(self):
        """Method to save best model depending on the monitored quantity."""
        val = self.exp.exp_logs.get(self.monitor)

        if self.improvement(score=val, best=self.best_val):

            self.checkpoint(epoch=self.exp.exp_logs.get(self.exp.epoch_key))

    def experiment_end(self):
        """Reset to default."""
        if self.mode == "max":
            self.best_val = -np.inf
        else:
            self.best_val = np.inf

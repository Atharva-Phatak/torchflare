"""Implements Model Checkpoint Callback."""
from abc import ABC
from typing import Dict

import numpy as np
import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class ModelCheckpoint(Callbacks, ABC):
    """Callback for Checkpointing your model."""

    def __init__(self, mode: str = "min", monitor: str = "val_loss"):
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
            3) 'scheduler_state_dict' : The state dictionary of scheduler.(If used)
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

        if self.mode == "max":
            self.best_val = -np.inf
            self.improvement = lambda val, best_val: val >= best_val + self.eps
        else:
            self.best_val = np.inf
            self.improvement = lambda val, best_val: val <= best_val + self.eps

    def checkpoint(self, epoch: int):
        """Method to save the state dictionaries of model, optimizer,etc.

        Args:
            epoch : The epoch at which model is saved.
        """
        if self.exp.scheduler_stepper is not None:
            torch.save(
                {
                    "model_state_dict": self.exp.model.state_dict(),
                    "optimizer_state_dict": self.exp.optimizer.state_dict(),
                    "scheduler_state_dict": self.exp.scheduler_stepper.scheduler.state_dict(),
                    "Epoch": epoch,
                },
                self.exp.path,
            )
        else:
            torch.save(
                {
                    "model_state_dict": self.exp.model.state_dict(),
                    "optimizer_state_dict": self.exp.optimizer.state_dict(),
                    "Epoch": epoch,
                },
                self.exp.path,
            )

    def epoch_end(self, epoch: int, logs: Dict):
        """Method to save best model depending on the monitored quantity.

        Args:
            epoch: The current epoch.
            logs: A dictionary containing metrics and loss values.
        """
        val = logs.get(self.monitor)

        if self.improvement(val=val, best_val=self.best_val):

            self.checkpoint(epoch=epoch)

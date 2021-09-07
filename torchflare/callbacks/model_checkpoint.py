"""Implements Model Checkpoint Callback."""
import os
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import torch

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.extra_utils import init_improvement
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


def get_state_dicts(d):
    """Get state dictionaries for input dictionary or nn_modules.

    Args:
        d : The input dict or nn_module.
    """
    if isinstance(d, dict):
        z = {k: v.state_dict() for k, v in d.items()}
    else:
        z = d.state_dict()
    return z


class ModelCheckpoint(Callbacks, ABC):
    """Callback for Checkpointing your model.

    Args:
            mode: One of {"min", "max"}.
                In min mode, training will stop when the quantity monitored has stopped decreasing
                in "max" mode it will stop when the quantity monitored has stopped increasing.
            monitor: The quantity to be monitored. (Default : val_loss)
                    If you want to monitor other metric just pass in the name of the metric.
            save_dir: The directory where you want to save the model files.
            file_name: The name of file. Default : model.bin

    Note:

             ModelCheckpoint will save state_dictionaries for model , optimizer , scheduler
             and the value of epoch with following key values:

            1) 'model_state_dict' : The state dictionary of model
            2) 'optimizer_state_dict'  : The state dictionary of optimizer

            Model checkpoint will be saved based \
            on the values of metrics/loss obtained from validation set.

    Raises:
        ValueError if monitor does not start with prefix ``val_`` or ``train_``.

    Example:
        .. code-block::

            import torchflare.callbacks as cbs
            model_ckpt = cbs.ModelCheckpoint(monitor="val_accuracy", mode="max")
    """

    def __init__(self, mode: str, monitor: str, save_dir: str = "./", file_name: str = "model.bin"):
        """Constructor for ModelCheckpoint class."""
        super(ModelCheckpoint, self).__init__(order=CallbackOrder.CHECKPOINT)
        if monitor.startswith("train_") or monitor.startswith("val_"):
            self.monitor = monitor
        else:
            raise ValueError("Monitor must have a prefix either train_ or val_.")
        self.mode = mode
        self.eps = 1e-7

        self.improvement, self.best_val = init_improvement(mode=self.mode, min_delta=self.eps)
        self.path = os.path.join(save_dir, file_name)

    def checkpoint(self, model, optimizer):
        """Method to save the state dictionaries of model, optimizer,etc."""
        state_dict = {
            "model_state_dict": get_state_dicts(d=model),
            "optimizer_state_dict": get_state_dicts(d=optimizer),
        }
        torch.save(state_dict, self.path)

    def on_epoch_end(self, experiment: "Experiment"):
        """Method to save best model depending on the monitored quantity."""
        val = experiment.exp_logs.get(self.monitor)

        if self.improvement(score=val, best=self.best_val):

            self.checkpoint(
                model=experiment.state.model,
                optimizer=experiment.state.optimizer,
            )

    def on_experiment_end(self, experiment: "Experiment"):
        """Reset to default."""
        if self.mode == "max":
            self.best_val = -np.inf
        else:
            self.best_val = np.inf

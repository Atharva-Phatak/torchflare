"""Implements Model History."""

from abc import ABC
from typing import TYPE_CHECKING

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class History(Callbacks, ABC):
    """Class to log metrics to console and save them to a CSV file."""

    def __init__(self):
        """Constructor class for History Class."""
        super(History, self).__init__(order=CallbackOrder.LOGGING)
        self.history = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Sets variables at experiment start."""
        self.history = {}

    def _update_history(self, logs):
        for key in logs:
            if key not in self.history:
                self.history[key] = []
                self.history[key].append(logs.get(key))
            else:
                self.history[key].append(logs.get(key))

    def on_epoch_end(self, experiment: "Experiment"):
        """Method to update history object at the end of every epoch."""
        self._update_history(logs=experiment.exp_logs)
        experiment.history = self.history

"""Implements Model History."""

from abc import ABC

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class History(Callbacks, ABC):
    """Class to log metrics to console and save them to a CSV file."""

    def __init__(self):
        """Constructor class for History Class."""
        super(History, self).__init__(order=CallbackOrder.LOGGING)
        self.history = None

    def experiment_start(self):
        """Sets variables at experiment start."""
        self.history = {}

    def _update_history(self, logs):
        for key in logs:
            if key not in self.history:
                self.history[key] = []
                self.history[key].append(logs.get(key))
            else:
                self.history[key].append(logs.get(key))

    def epoch_end(self):
        """Method to update history object at the end of every epoch."""
        self._update_history(logs=self.exp.exp_logs)
        self.exp.history = self.history

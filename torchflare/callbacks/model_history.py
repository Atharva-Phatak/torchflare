"""Implements Model History."""
import os
from abc import ABC
from typing import Dict

import pandas as pd

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
                self.history[key].append(logs[key])
            else:
                self.history[key].append(logs[key])

    def _store_history(self):

        path = os.path.join(self.exp.save_dir, "history.csv")
        df = pd.DataFrame.from_dict(self.history)
        df.to_csv(path, index=False)

    def epoch_end(self, epoch: int, logs: Dict):
        """Method to update history object at the end of every epoch.

        Args:
            epoch: The current epoch.
            logs: Dictionary containing the metric and loss values.1
        """
        self._update_history(logs=dict(Epoch=epoch, **logs))

    def experiment_end(self):
        """Method to store history object at experiment enc."""
        self._store_history()

"""Implementation of Early stopping."""
from abc import ABC
from typing import Dict

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class EarlyStopping(Callbacks, ABC):
    """Implementation of Early Stopping Callback."""

    def __init__(
        self, monitor: str = "val_loss", patience: int = 5, mode: str = "min", min_delta: float = 1e-7,
    ):
        """Constructor for EarlyStopping class.

        Args:
            monitor: The quantity to be monitored. (Default : val_loss)
                    If you want to monitor other metric just pass in the name of the metric.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored
                has stopped decreasing.In "max" mode it will stop when the quantity monitored has stopped increasing.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.

        Note:

            EarlyStopping will only use the values of metrics/loss obtained on validation set.
        """
        super(EarlyStopping, self).__init__(order=CallbackOrder.STOPPING)

        if "val_" not in monitor:
            self.monitor = "val_" + monitor
        else:
            self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.stopping_counter = 0
        self.best_score = None
        self.improvement = None
        if self.mode == "min":

            self.improvement = lambda score, best_score: score <= (best_score - self.min_delta)

        else:

            self.improvement = lambda score, best_score: score >= (best_score + self.min_delta)

        self.stopping_counter = 0

    def epoch_end(self, epoch: int, logs: Dict):
        """Function which will determine when to stop the training depending on the score.

        Args:
            logs: A dictionary containing metrics and loss values.
            epoch : The current epoch
        """
        epoch_score = logs.get(self.monitor)
        if self.best_score is None or self.improvement(score=epoch_score, best_score=self.best_score):

            self.best_score = epoch_score

        else:

            self.stopping_counter += 1

        if self.stopping_counter >= self.patience:
            print("Early Stopping !")
            self.exp.stop_training = True

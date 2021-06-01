"""Implementation of Early stopping."""
import math
from abc import ABC
from typing import TYPE_CHECKING

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.extra_utils import init_improvement
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class EarlyStopping(Callbacks, ABC):
    """Implementation of Early Stopping Callback.

    Args:
            monitor: The quantity to be monitored. (Default : val_loss)
                    If you want to monitor other metric just pass in the name of the metric.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored
                has stopped decreasing.In "max" mode it will stop when the quantity monitored has stopped increasing.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.

    Note:

        EarlyStopping will only use the values of metrics/loss obtained on validation set.

    Example:
        .. code-block:: python

            import torchflare.callbacks as cbs
            early_stop = cbs.EarlyStopping(monitor="val_accuracy", patience=5, mode="max")

    """

    def __init__(
        self,
        mode: str,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-7,
    ):
        """Constructor for EarlyStopping class."""
        super(EarlyStopping, self).__init__(order=CallbackOrder.STOPPING)

        if "val_" not in monitor:
            self.monitor = "val_" + monitor
        else:
            self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.stopping_counter = 0
        self.improvement, self.best_score = init_improvement(mode=self.mode, min_delta=self.min_delta)

        self.stopping_counter = 0

    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
        self.stopping_counter = 0
        self.best_score = math.inf if self.mode == "min" else -math.inf

    def on_epoch_end(self, experiment: "Experiment"):
        """Function which will determine when to stop the training depending on the score."""
        epoch_score = experiment.exp_logs.get(self.monitor)
        if self.improvement(epoch_score, self.best_score):
            self.best_score = epoch_score
            self.stopping_counter = 0

        else:
            self.stopping_counter += 1
            if self.stopping_counter >= self.patience:
                print("Early Stopping !")
                experiment.stop_training = True

    def on_experiment_end(self, experiment: "Experiment"):
        """Reset to defaults."""
        self.stopping_counter = 0
        self.best_score = None
        self.improvement = None

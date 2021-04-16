"""Implements Tensorboard Logger."""
from abc import ABC

from torch.utils.tensorboard import SummaryWriter

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class TensorboardLogger(Callbacks, ABC):
    """Callback to use Tensorboard to log your metrics and losses."""

    def __init__(self, log_dir: str):
        """Constructor for TensorboardLogger class.

        Args:
            log_dir: The directory where you want to save your experiments.
        """
        super(TensorboardLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.log_dir = log_dir
        self._experiment = SummaryWriter(log_dir=self.log_dir)

    def epoch_end(self, epoch: int, logs: dict):
        """Method to log metrics and values at the end of very epoch.

        Args:
            logs: A dictionary containing metrics and loss values.
            epoch: The current epoch
        """
        for key, value in logs.items():
            if not isinstance(value, str):
                self._experiment.add_scalar(tag=key, scalar_value=value, global_step=epoch)

    def experiment_end(self):
        """Method to end experiment after training is done."""
        self._experiment.close()

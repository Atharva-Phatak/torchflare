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
        self._experiment = None

    def experiment_start(self):
        """Start of experiment."""
        self._experiment = SummaryWriter(log_dir=self.log_dir)

    def epoch_end(self):
        """Method to log metrics and values at the end of very epoch."""
        for key, value in self.exp.exp_logs.items():
            if key != self.exp.epoch_key:
                epoch = self.exp.exp_logs[self.exp.epoch_key]
                self._experiment.add_scalar(tag=key, scalar_value=value, global_step=epoch)

    def experiment_end(self):
        """Method to end experiment after training is done."""
        self._experiment.close()
        self._experiment = None

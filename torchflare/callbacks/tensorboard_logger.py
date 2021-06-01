"""Implements Tensorboard Logger."""
from abc import ABC
from typing import TYPE_CHECKING

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder
from torchflare.utils.imports_check import module_available

_AVAILABLE = module_available("tensorboard")
if _AVAILABLE:
    from torch.utils.tensorboard import SummaryWriter
else:
    SummaryWriter = None


if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class TensorboardLogger(Callbacks, ABC):
    """Callback to use Tensorboard to log your metrics and losses.

    Args:
            log_dir: The directory where you want to save your experiments.
    """

    def __init__(self, log_dir: str):
        """Constructor for TensorboardLogger class."""
        super(TensorboardLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.log_dir = log_dir
        self._experiment = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
        self._experiment = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, experiment: "Experiment"):
        """Method to log metrics and values at the end of very epoch."""
        for key, value in experiment.exp_logs.items():
            if key != experiment.epoch_key:
                epoch = experiment.exp_logs[experiment.epoch_key]
                self._experiment.add_scalar(tag=key, scalar_value=value, global_step=epoch)

    def on_experiment_end(self, experiment: "Experiment"):
        """Method to end experiment after training is done."""
        self._experiment.close()
        self._experiment = None

from typing import TYPE_CHECKING

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class AvgLoss(Callbacks):
    """Class for averaging the loss."""

    def __init__(self):
        super(AvgLoss, self).__init__(order=CallbackOrder.LOSS)
        self.accum_loss, self.count = {}, 0
        self.reset()

    def reset(self):
        """Reset the variables."""
        self.accum_loss, self.count = {}, 0

    def on_batch_end(self, experiment: "Experiment"):
        """Accumulate values."""
        bs = experiment.state.dataloaders.get(experiment.which_loader).batch_size
        for k, v in experiment.loss_per_batch.items():
            if k not in self.accum_loss:
                self.accum_loss[k] = v * bs
            else:
                self.accum_loss[k] += v * bs
        self.count += bs

    def on_loader_end(self, experiment: "Experiment"):
        """Method to return computed dictionary."""
        prefix = experiment.get_prefix()
        loss_dict = {prefix + k: v / self.count for k, v in self.accum_loss.items()}
        self.reset()
        experiment.monitors[experiment.which_loader] = loss_dict

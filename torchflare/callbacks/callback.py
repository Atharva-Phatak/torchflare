"""Implementation of Callbacks and CallbackRunner."""
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


def sort_callbacks(callbacks: List) -> List:
    """Method to sort callbacks.

    Args:
        callbacks: List of callbacks.

    Returns:
        Callbacks sorted according to order.
    """
    callbacks = sorted(callbacks, key=lambda cb: cb.order.value)
    return callbacks


class Callbacks:
    """Simple class for defining callbacks depending on the experiment state."""

    def __init__(self, order):
        """Constructor for class callbacks.

        Args:
            order: The priority value for callbacks so that \
            they can be sorted according to the value.
        """
        self.order = order

    def on_batch_start(self, experiment: "Experiment"):
        """Start of batch."""
        raise NotImplementedError

    def on_batch_end(self, experiment: "Experiment"):
        """End of Batch."""
        raise NotImplementedError

    # skipcq: PTC-W0049
    def on_loader_start(self, experiment: "Experiment"):
        """Start of loader."""
        raise NotImplementedError

    def on_loader_end(self, experiment: "Experiment"):
        """End of loader."""
        raise NotImplementedError

    def on_epoch_start(self, experiment: "Experiment"):
        """Start of Epoch."""
        raise NotImplementedError

    def on_epoch_end(self, experiment: "Experiment"):
        """End of epoch."""
        raise NotImplementedError

    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
        raise NotImplementedError

    def on_experiment_end(self, experiment: "Experiment"):
        """End of experiment."""
        raise NotImplementedError

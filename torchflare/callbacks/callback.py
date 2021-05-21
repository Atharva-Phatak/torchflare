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
            order: The priority value for callbacks so that they can be sorted according to the value.
        """
        self.order = order

    # skipcq: PTC-W0049
    def on_batch_start(self, experiment: "Experiment"):
        """Start of batch."""
        # skipcq: PYL-W0107
        pass

    # skipcq: PTC-W0049
    def on_batch_end(self, experiment: "Experiment"):
        """End of Batch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_loader_start(self, experiment: "Experiment"):
        """Start of loader."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_loader_end(self, experiment: "Experiment"):
        """End of loader."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_epoch_start(self, experiment: "Experiment"):
        """Start of Epoch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_epoch_end(self, experiment: "Experiment"):
        """End of epoch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def on_experiment_end(self, experiment: "Experiment"):
        """End of experiment."""
        pass  # skipcq: PYL-W0107

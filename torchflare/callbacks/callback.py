"""Implementation of Callbacks and CallbackRunner."""
from enum import Enum
from typing import List


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
        self.exp = None
        self.order = order

    def set_experiment(self, exp):  # noqa
        self.exp = exp

    # skipcq: PTC-W0049
    def batch_start(self):
        """Start of batch."""
        # skipcq: PYL-W0107
        pass

    # skipcq: PTC-W0049
    def batch_end(self):
        """End of Batch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def loader_start(self):
        """Start of loader."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def loader_end(self):
        """End of loader."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def epoch_start(self):
        """Start of Epoch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def epoch_end(self):
        """End of epoch."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def experiment_start(self):
        """Start of experiment."""
        pass  # skipcq: PYL-W0107

    # skipcq: PTC-W0049
    def experiment_end(self):
        """End of experiment."""
        pass  # skipcq: PYL-W0107


class CallbackRunner:
    """Class to run all the callbacks."""

    def __init__(self, callbacks):
        """Constructor for CallbackRunner Class.

        Args:
            callbacks: The List of callbacks
        """
        self.callbacks = callbacks

    def set_experiment(self, exp):  # noqa
        for cb in self.callbacks:
            cb.set_experiment(exp)

    def __call__(self, current_state: Enum):
        """Runs callbacks depending on the current experiment state."""
        for cb in self.callbacks:
            try:
                _ = getattr(cb, current_state.value)()
            except AttributeError:
                pass

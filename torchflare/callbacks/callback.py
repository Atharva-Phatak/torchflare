"""Implementation of Callbacks and CallbackRunner."""
from enum import Enum
from typing import Dict, List

from torchflare.callbacks.states import ExperimentStates


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

    def epoch_start(self, epoch, logs):
        """Start of Epoch."""
        raise NotImplementedError

    def epoch_end(self, epoch, logs):
        """End of epoch."""
        raise NotImplementedError

    def experiment_start(self):
        """Start of experiment."""
        raise NotImplementedError

    def experiment_end(self):
        """End of experiment."""
        raise NotImplementedError


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

    def __call__(self, current_state: Enum, epoch: int = None, logs: Dict = None):
        """Runs callbacks depending on the current experiment state.

        Args:
            current_state : The current model state while training
            logs : A dict containing all the metrics and loss values
            epoch: The current epoch.
        """
        for cb in self.callbacks:
            if current_state.value in list(cb.__class__.__dict__):
                if current_state in (ExperimentStates.EXP_END, ExperimentStates.EXP_START):
                    _ = getattr(cb, current_state.value)()
                else:
                    _ = getattr(cb, current_state.value)(epoch=epoch, logs=logs)

"""Implements TimeCallback."""
import time

from fastprogress.fastprogress import format_time

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class TimeCallback(Callbacks):
    """Callback to measure time per epoch."""

    def __init__(self):
        """Constructor method for TimeCallback."""
        super(TimeCallback, self).__init__(order=CallbackOrder.INTERNAL)
        self.time = None

    def epoch_start(self, epoch, logs):
        """Method ot start measuring time at the start of epoch."""
        self.time = time.time()

    def epoch_end(self, epoch, logs):
        """Method to stop measuring time at the end of epoch and format it."""
        end = time.time() - self.time
        end = format_time(end)
        self.exp.exp_logs.update({"Time": end})


__all__ = ["TimeCallback"]

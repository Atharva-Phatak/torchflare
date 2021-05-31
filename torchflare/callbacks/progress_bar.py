"""Implementation of Progress Bar."""
import math
import sys
import time
from typing import TYPE_CHECKING, Dict

from torch.utils.data import DataLoader

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment

# Adapted From: https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/utils/generic_utils.py


class ProgressBar(Callbacks):
    """Class to create almost keras like progress bar."""

    def __init__(
        self,
        width: int = 25,
        interval: float = 0.05,
        unit_name: str = "step",
    ):
        """Constructor class for ProgressBar."""
        super(ProgressBar, self).__init__(order=CallbackOrder.EXTERNAL)
        self.num_epochs = None
        self.width = width
        self.interval = interval
        self.unit_name = unit_name

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
        )

        self._total_width = 0
        self._seen_so_far = 0
        self._start = time.time()
        self._last_update = 0
        self.num_steps = 0

    def on_experiment_start(self, experiment: "Experiment"):
        """On start of experiment."""
        self.num_epochs = experiment.num_epochs

    def on_epoch_start(self, experiment: "Experiment"):
        """On start of epoch."""
        sys.stdout.write("\n")
        if (experiment.current_epoch is not None) and (self.num_epochs is not None):
            sys.stdout.write(f"Epoch: {experiment.current_epoch}/{self.num_epochs}")
            sys.stdout.write("\n")

    def _create_bar(self, stage: str, current_step: int):
        """Method to create the progress bar.

        Args:
            current_step: The current iteration step.
            stage: The current stage of the experiment.
        """
        bar = f"{stage}: {current_step}/{self.num_steps} ["
        prog = float(current_step) / self.num_steps
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += "=" * (prog_width - 1)
            if current_step < self.num_steps:
                bar += ">"
            else:
                bar += "="
            bar += "." * (self.width - prog_width)
            bar += "]"

        return bar

    def _update(self, stage: str, current_step: int, values: Dict[str, float]):
        """Update progress bar every step.

        Args:
            current_step: The current step of iteration.
            values: A dictionary containing the keys , values for metrics/loss to be displayed.
            stage: The current stage whether Train or validation.
        """
        self._seen_so_far = current_step
        now = time.time()
        info = f"- {now - self._start:.0f}s"
        if now - self._last_update < self.interval and self.num_steps is not None and current_step < self.num_steps:
            return

        prev_total_width = self._total_width
        if self._dynamic_display:
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")

        bar = self._create_bar(stage=stage, current_step=current_step)
        self._total_width = len(bar)
        sys.stdout.write(bar)

        if current_step:
            time_per_unit = (now - self._start) / current_step
        else:
            time_per_unit = 0
        if self.num_steps is not None and current_step < self.num_steps:
            eta = time_per_unit * (self.num_steps - current_step)
            if eta > 3600:
                eta_format = "%d:%02d:%02d" % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = "%d:%02d" % (eta // 60, eta % 60)
            else:
                eta_format = "%ds" % eta

            info = f" - ETA: {eta_format}"

        else:
            if time_per_unit >= 1 or time_per_unit == 0:
                info += " {:.0f}s/{}".format(time_per_unit, self.unit_name)
            elif time_per_unit >= 1e-3:
                info += " {:.0f}ms/{}".format(time_per_unit * 1e3, self.unit_name)
            else:
                info += " {:.0f}us/{}".format(time_per_unit * 1e6, self.unit_name)

        for k, v in values.items():

            info += f" - {k}:"
            if abs(v) > 1e-3:
                info += f" {v:.4f}"
            else:
                info += f" {v:.4e}"

        self._total_width += len(info)
        if prev_total_width > self._total_width:
            info += " " * (prev_total_width - self._total_width)

        if self.num_steps is not None and current_step >= self.num_steps:
            info += "\n"

        sys.stdout.write(info)
        sys.stdout.flush()
        self._last_update = now

    def on_batch_end(self, experiment: "Experiment"):
        """On end of a batch."""
        values = {"loss": experiment.loss.item()}
        self._update(current_step=experiment.batch_idx, values=values, stage=experiment.stage)

    def on_loader_end(self, experiment: "Experiment"):
        """On end of dataloader."""
        self._update(
            current_step=self._seen_so_far + 1, values=experiment.monitors[experiment.stage], stage=experiment.stage
        )
        self.reset()

    # noinspection PyTypeChecker
    @staticmethod
    def calculate_steps(dl: DataLoader):
        """Calculate num of steps.

        Args:
            dl : The dataloader.

        Returns:
            The number of steps.
        """
        steps = len(dl.dataset) / dl.batch_size
        return math.ceil(steps)

    def on_loader_start(self, experiment: "Experiment"):
        """On start of loader.."""
        dl = experiment.dataloaders.get(experiment.stage)
        self.num_steps = self.calculate_steps(dl=dl)

    def reset(self):
        """Method to reset internal variables."""
        self._total_width = 0
        self._seen_so_far = 0
        self._start = time.time()
        self._last_update = 0

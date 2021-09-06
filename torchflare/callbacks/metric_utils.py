"""Implements container for loss and metric computation."""
from typing import TYPE_CHECKING, Dict, List

from torchmetrics import MetricCollection

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


def detach_tensor(x):
    return x.detach().cpu()


class MetricCallback(Callbacks):
    """Class to run and compute Loss and Metrics."""

    def __init__(self, metrics: List = None):
        """Constructor class for MetricContainer.

        Args:
            metrics: The list of metrics
        """
        super(MetricCallback, self).__init__(CallbackOrder.METRICS)
        metrics = MetricCollection(metrics)
        self.metrics = {
            "train": metrics.clone(prefix="train_"),
            "eval": metrics.clone(prefix="val_"),
        }

    def reset(self, stage) -> None:
        """Method to reset the state of metrics and loss meter."""
        self.metrics[stage].reset()

    def on_loader_end(self, experiment: "Experiment") -> None:
        """Method to compute the metrics once accumulation of values is done."""
        metric_dict = self.compute(stage=experiment.which_loader)
        self.reset(stage=experiment.which_loader)
        experiment.monitors[experiment.which_loader].update(metric_dict)

    def on_batch_end(self, experiment: "Experiment") -> None:
        """Accumulate values."""
        preds = detach_tensor(x=experiment.batch_outputs[experiment.prediction_key])
        targets = detach_tensor(x=experiment.batch[experiment.target_key])
        self.metrics[experiment.which_loader].update(preds, targets)

    def compute(self, stage) -> Dict:
        """Compute values."""
        return {k.lower(): v.item() for k, v in self.metrics[stage].compute().items()}


__all__ = ["MetricCallback"]

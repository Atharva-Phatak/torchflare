"""Implements container for loss and metric computation."""
from typing import TYPE_CHECKING, Dict, List

from torchflare.experiments.simple_utils import wrap_metric_names

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class MetricContainer:
    """Class to run and compute Loss and Metrics."""

    def __init__(self, metrics: List = None):
        """Constructor class for MetricContainer.

        Args:
            metrics: The list of metrics
        """
        if metrics is not None:
            self.metrics = metrics
            self.metric_names = wrap_metric_names(metric_list=metrics)
        else:
            self.metrics = None

        self.prefix = None
        self.metric_dict = None

    def reset(self):
        """Method to reset the state of metrics and loss meter."""
        _ = map(lambda x: x.reset(), self.metrics)

    def value(self, experiment: "Experiment") -> Dict:
        """Method to compute the metrics once accumulation of values is done.

        Returns:
            A dictionary containing corresponding metrics.
        """
        self.compute(experiment=experiment)
        self.reset()
        return self.metric_dict

    def accumulate(self, experiment: "Experiment"):
        """Accumulate values."""
        for metric in self.metrics:
            metric.accumulate(experiment.preds, experiment.y)

    def compute(self, experiment: "Experiment"):
        """Compute values."""
        self.prefix = experiment.get_prefix()
        self.compute_vals()

    def compute_vals(self):
        """Compute values."""
        metric_vals = list(map(lambda x: x.value.item(), self.metrics))
        self.metric_dict = {self.prefix + key: value for key, value in zip(self.metric_names, metric_vals)}


__all__ = ["MetricContainer"]

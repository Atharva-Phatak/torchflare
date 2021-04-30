"""Implements container for loss and metric computation."""
from typing import Dict, List

from torchflare.experiments.simple_utils import wrap_metric_names


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

        self.exp = None
        self.prefix = None
        self.metric_dict = None

    def set_experiment(self, exp):  # noqa
        self.exp = exp

    def reset(self):
        """Method to reset the state of metrics and loss meter."""
        if self.exp.compute_metric_flag:
            _ = map(lambda x: x.reset(), self.metrics)

    @property
    def value(self) -> Dict:
        """Method to compute the metrics once accumulation of values is done.

        Returns:
            A dictionary containing corresponding metrics.
        """
        self.compute()
        self.reset()
        return self.metric_dict

    def accumulate(self):
        """Method to accumulate output of every batch."""
        self.accum_vals()

    def accum_vals(self):
        """Accumulate values."""
        if self.exp.compute_metric_flag:
            for metric in self.metrics:
                metric.accumulate(self.exp.preds, self.exp.y)

    def compute(self):
        """Compute values."""
        self.prefix = self.exp.get_prefix()
        self.compute_vals()

    def compute_vals(self):
        """Compute values."""
        if self.exp.compute_metric_flag:
            metric_vals = list(map(lambda x: x.value.item(), self.metrics))
            self.metric_dict = {self.prefix + key: value for key, value in zip(self.metric_names, metric_vals)}


__all__ = ["MetricContainer"]

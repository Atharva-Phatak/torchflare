"""Implements container for loss and metric computation."""
from typing import Dict, List

from torchflare.experiments.simple_utils import wrap_metric_names
from torchflare.utils.average_meter import AverageMeter


class MetricAndLossContainer:
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
        self.loss_meter = AverageMeter()

    def set_experiment(self, exp):  # noqa
        self.exp = exp

    def reset(self):
        """Method to reset the state of metrics and loss meter."""
        self.loss_meter.reset()
        if self.metrics is not None and self.exp.compute_metric_flag:

            for metric in self.metrics:
                metric.reset()

    def compute(self, prefix: str) -> Dict:
        """Method to compute the metrics once accumulation of values is done.

        Args:
            prefix : The prefix to add to the keys.

        Returns:
            A dictionary containing corresponding metrics.
        """
        metric_dict = {prefix + "loss": self.loss_meter.avg}
        if self.metrics is not None and self.exp.compute_metric_flag:
            metric_vals = [metric.compute().item() for metric in self.metrics]
            metric_dict.update({prefix + key: value for key, value in zip(self.metric_names, metric_vals)})

        return metric_dict

    def accumulate(self, loss, n, op=None, y=None):
        """Method to accumulate output of every batch.

        Args:
            op : The output of the net.
            y : The corresponding targets/targets.
            loss : the loss per batch
            n : The number of batches in a single iteration(batch_size)
        """
        self.loss_meter.update(loss, n)
        if self.metrics is not None and self.exp.compute_metric_flag:
            for metric in self.metrics:
                metric.accumulate(op, y)


__all__ = ["MetricAndLossContainer"]

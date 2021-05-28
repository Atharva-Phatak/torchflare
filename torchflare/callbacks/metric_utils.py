"""Implements container for loss and metric computation."""
from typing import TYPE_CHECKING, Dict, List

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


def wrap_metric_names(metric_list: List):
    """Method to  get the metric names from metric list.

    Args:
        metric_list : A list of metrics.

    Returns:
        list of metric name.s
    """
    return [metric.handle() for metric in metric_list]


class MetricCallback(Callbacks):
    """Class to run and compute Loss and Metrics."""

    def __init__(self, metrics: List = None):
        """Constructor class for MetricContainer.

        Args:
            metrics: The list of metrics
        """
        super(MetricCallback, self).__init__(CallbackOrder.METRICS)
        if metrics is not None:
            self.metrics = metrics
            self.metric_names = wrap_metric_names(metric_list=metrics)
        else:
            self.metrics = None
        self.metric_dict = None

    def reset(self):
        """Method to reset the state of metrics and loss meter."""
        _ = map(lambda x: x.reset(), self.metrics)

    def on_loader_end(self, experiment: "Experiment") -> Dict:
        """Method to compute the metrics once accumulation of values is done."""
        prefix = experiment.get_prefix()
        self.compute(prefix)
        self.reset()
        experiment.monitors[experiment.stage].update(self.metric_dict)

    def on_batch_end(self, experiment: "Experiment"):
        """Accumulate values."""
        for metric in self.metrics:
            metric.accumulate(experiment.preds, experiment.y)

    def compute(self, prefix):
        """Compute values."""
        metric_vals = list(map(lambda x: x.value.item(), self.metrics))
        self.metric_dict = {prefix + key: value for key, value in zip(self.metric_names, metric_vals)}


__all__ = ["MetricCallback"]

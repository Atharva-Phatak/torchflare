# flake8: noqa
import torch

from torchflare.metrics.accuracy_meter import Accuracy
from torchflare.metrics.metric_utils import MetricContainer
from torchflare.metrics.precision_meter import Precision

torch.manual_seed(42)


class Experiment:
    def __init__(self, train=True, metrics=None):

        self.train_metric = train
        self.val_metrics = True
        self._metric_runner = MetricContainer(metrics=metrics)
        self.compute_metric_flag = None
        self.preds = torch.randn(100, 1)
        self.y = torch.randint(0, 2, size=(100,))
        self.is_training = False

    def get_prefix(self):
        return "train_" if self.is_training else "val_"

    def train_fn(self):

        self.is_training = True
        self._metric_runner.reset()
        self.compute_metric_flag = self.train_metric
        loss = 10
        for _ in range(10):

            loss = loss * 0.1
            self._metric_runner.accumulate(self)

        metrics = self._metric_runner.value(self)
        assert isinstance(metrics, dict) is True
        loss_bool = "train_accuracy" in metrics
        assert loss_bool is True

    def val_fn(self):

        self.is_training = False
        self.compute_metric_flag = self.val_metrics
        self._metric_runner.reset()
        loss = 20
        for _ in range(10):
            loss = loss * 0.1
            self._metric_runner.accumulate(self)

        metrics = self._metric_runner.value(self)
        assert isinstance(metrics, dict) is True
        loss_bool = "val_accuracy" in metrics
        assert loss_bool is True

    def fit(self):

        for _ in range(1):

            self.train_fn()
            self.val_fn()


def test():
    metrics = [
        Accuracy(num_classes=1, threshold=0.5, multilabel=False),
        Precision(num_classes=1, threshold=0.5, multilabel=False, average="macro",),
    ]
    d = Experiment(train=True, metrics=metrics)
    d.fit()

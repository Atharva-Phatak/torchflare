# flake8: noqa
import torch

from torchflare.metrics import Accuracy, MetricAndLossContainer, Precision

torch.manual_seed(42)
outputs = torch.randn(100, 1)
targets = torch.randint(0, 2, size=(100,))


class DummyPipe:
    def __init__(self, train=True, metrics=None):

        self.train_metric = train
        self.val_metrics = True
        self._metric_runner = MetricAndLossContainer(metrics=metrics)
        self._metric_runner._set_experiment(self)
        self._compute_metric_flag = None

    def train_fn(self):

        self._metric_runner.reset()
        self._compute_metric_flag = self.train_metric
        loss = 10
        for _ in range(10):

            loss = loss * 0.1
            self._metric_runner.accumulate(op=outputs, y=targets, loss=loss, n=1)

        metrics = self._metric_runner.compute(prefix="train_")
        loss_bool = "train_loss" in metrics
        assert loss_bool == True

    def val_fn(self):

        self._compute_metric_flag = self.val_metrics
        self._metric_runner.reset()
        loss = 20
        for _ in range(10):
            loss = loss * 0.1
            self._metric_runner.accumulate(op=outputs, y=targets, loss=loss, n=1)

        metrics = self._metric_runner.compute(prefix="val_")
        loss_bool = "val_loss" in metrics
        assert loss_bool == True

    def fit(self):

        for _ in range(1):

            self.train_fn()
            self.val_fn()


def test():
    metrics = [
        Accuracy(num_classes=1, threshold=0.5, multilabel=False),
        Precision(num_classes=1, threshold=0.5, multilabel=False, average="macro",),
    ]
    d = DummyPipe(train=True, metrics=metrics)
    d.fit()

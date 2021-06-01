# flake8: noqa
import torch

from torchflare.metrics.accuracy_meter import Accuracy
from torchflare.callbacks.metric_utils import MetricCallback
from torchflare.metrics.precision_meter import Precision

torch.manual_seed(42)


class Experiment:
    def __init__(self, train=True, metrics=None):

        self.train_metric = train
        self.val_metrics = True
        self._metric_runner = MetricCallback(metrics=metrics)
        self.preds = torch.randn(100, 1)
        self.y = torch.randint(0, 2, size=(100,))
        self.monitors = {"Train" : {} , "Valid" : {}}
        self.stage = None
        self.is_training = False

    def get_prefix(self):
        if self.stage == "Train":
            return "train_"
        else:
            return "val_"
    def train_fn(self):

        self.stage = "Train"
        self._metric_runner.on_experiment_start(self)
        loss = 10
        for _ in range(10):

            loss = loss * 0.1
            self._metric_runner.on_batch_end(self)

        self._metric_runner.on_loader_end(self)
        #assert isinstance(metric, dict) is True
        loss_bool = "train_accuracy" in self.monitors[self.stage]
        assert loss_bool is True

    def val_fn(self):

        self.stage = "Valid"
        self._metric_runner.on_experiment_start(self)
        loss = 20
        for _ in range(10):
            loss = loss * 0.1
            self._metric_runner.on_batch_end(self)

        self._metric_runner.on_loader_end(self)
        loss_bool = "val_accuracy" in self.monitors[self.stage]
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

# flake8: noqa
import torchmetrics
from torchflare.callbacks.metric_utils import MetricCallback
import torch


class Experiment:
    def __init__(self, metrics):

        self.metric_callback = MetricCallback(metrics=metrics)
        self.prediction_key = "prediction"
        self.target_key = "targets"
        self.batch_outputs = {self.prediction_key: torch.tensor([0, 2, 1, 3])}
        self.batch = {self.target_key: torch.tensor([0, 1, 2, 3])}
        self.monitors = {"train": {}, "eval": {}}
        self.which_loader = None

    def train_fn(self):

        self.which_loader = "train"
        for _ in range(10):
            self.metric_callback.on_batch_end(self)

        self.metric_callback.on_loader_end(self)
        # assert isinstance(metric, dict) is True
        metric_bool = "train_accuracy" in self.monitors[self.which_loader]
        assert metric_bool is True

    def val_fn(self):

        self.which_loader = "eval"
        loss = 20
        for _ in range(10):
            loss = loss * 0.1
            self.metric_callback.on_batch_end(self)

        self.metric_callback.on_loader_end(self)
        metric_bool = "val_accuracy" in self.monitors[self.which_loader]
        assert metric_bool is True

    def fit(self):

        for _ in range(1):
            self.train_fn()
            self.val_fn()


def test_metric_callback():
    metrics = [torchmetrics.Accuracy()]
    exp = Experiment(metrics=metrics)
    exp.fit()

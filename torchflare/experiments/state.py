"""Implements Base State."""
import os
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

import torchflare.metrics.metric_utils as metric_utils
from torchflare.callbacks.callback import CallbackRunner, sort_callbacks
from torchflare.callbacks.model_history import History
from torchflare.callbacks.progress_bar import ProgressBar
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.simple_utils import AvgLoss, numpy_to_torch
from torchflare.utils.seeder import seed_all


class BaseState:
    """Class to set user and internal variables along with some utility functions."""

    def __init__(
        self,
        num_epochs: int,
        fp16: bool,
        device: str,
        seed: int = 42,
    ):
        """Init method to set up important variables for training and validation.

        Args:
            num_epochs : The number of epochs to save model.
            fp16 : Set this to True if you want to use mixed precision training(Default : False)
            device : The device where you want train your model.
            seed: The seed to ensure reproducibility.

        Note:
            If batch_mixers are used then set compute_train_metrics to False.
            Also, only validation metrics will be computed if special batch_mixers are used.
        """
        self.num_epochs = num_epochs
        self.fp16 = fp16
        self.device = device
        self.seed = seed
        self.train_key, self.val_key, self.epoch_key = "train_", "val_", "Epoch"

        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        self.model = None
        self.resume_checkpoint = None
        self.main_metric = None
        self.stop_training = None
        self.experiment_state = None
        self._callback_runner = None
        self.optimizer = None
        self.criterion = None

        self.exp_logs = None
        self.history = None
        self._train_monitor, self._val_monitor = None, None
        self._metric_runner = None
        self.train_dl, self.valid_dl = None, None
        self.loss, self.loss_meter = None, None
        self.x, self.y = None, None
        self.preds = None
        self.is_training = None
        self.metrics = None
        self.batch_idx, self.current_epoch = None, 0
        self._step, self._iterator = None, None
        self.plot_dir = "./plots"

    def get_prefix(self):
        """Generates the prefix for training and validation.

        Returns:
            The prefix for training or validation.
        """
        return self.train_key if self.is_training else self.val_key

    def _set_callbacks(self, callbacks: List):
        default_callbacks = [ProgressBar(), History()]
        if callbacks is not None:
            default_callbacks.extend(callbacks)

        default_callbacks = sort_callbacks(default_callbacks)
        self._callback_runner = CallbackRunner(callbacks=default_callbacks)
        self._callback_runner.set_experiment(self)

    def _set_metrics(self, metrics):

        if metrics is not None:
            self._metric_runner = metric_utils.MetricContainer(metrics=metrics)
            self._metric_runner.set_experiment(self)

    def _set_params(self, optimizer_params):

        if "model_params" in optimizer_params:
            grad_params = optimizer_params.pop("model_params")
        else:
            grad_params = (param for param in self.model.parameters() if param.requires_grad)

        return grad_params

    def _set_optimizer(self, optimizer, optimizer_params):

        grad_params = self._set_params(optimizer_params=optimizer_params)
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)(grad_params, **optimizer_params)
        else:
            self.optimizer = optimizer

    def _set_criterion(self, criterion):

        self.criterion = get_criterion(criterion=criterion)
        self.loss_meter = AvgLoss()
        self.loss_meter.set_experiment(self)

    @staticmethod
    def _dataloader_from_data(args, dataloader_kwargs):
        args = numpy_to_torch(args)
        dataset = TensorDataset(*args) if len(args) > 1 else args[0]
        return DataLoader(dataset, **dataloader_kwargs)

    def _model_to_device(self):
        """Function to move model to device."""
        if next(self.model.parameters()).is_cuda is False:
            self.model.to(self.device)

    def _reset_model_logs(self):
        if bool(self.exp_logs):
            self.exp_logs = None

    def initialise(self):
        """Method initialise some stuff."""
        seed_all(self.seed)
        self._model_to_device()
        self._reset_model_logs()

    def cleanup(self):
        """Method Cleanup internal variables."""
        self._train_monitor, self._val_monitor, self.exp_logs = None, None, {}
        self.experiment_state = None
        self.loss, self.x, self.y, self.preds = None, None, None, None
        self.batch_idx, self.current_epoch = None, None
        self._step, self._iterator = None, None

    def _create_plots(self, key):
        for k, v in self.history.items():
            if key in k:
                plt.plot(self.history.get(self.epoch_key), v, "-o", label=k)
        plt.title(f"{key.upper()}/{self.epoch_key.upper()}", fontweight="bold")
        plt.ylabel(f"{key.upper()}", fontweight="bold")
        plt.xlabel(self.epoch_key.upper(), fontweight="bold")
        plt.grid(True)
        plt.legend(loc="upper left")

    def _save_fig(self, save: bool, key: str):

        if save is not None:
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
            save_path = os.path.join(self.plot_dir, f"{key}-vs-{self.epoch_key.lower()}.jpg")
            plt.savefig(save_path, dpi=150)

    def plot_history(self, keys: List[str], plot_fig: bool = True, save_fig: bool = False):
        """Method to plot model history.

        Args:
            keys: A key value in lower case. Ex accuracy or loss
            save_fig: Set to True if you want to save_fig.
            plot_fig: Whether to plot the figure.
        """
        for key in keys:
            plt.style.use("seaborn")
            f = plt.figure()
            self._create_plots(key=key)
            self._save_fig(save=save_fig, key=key)
            if plot_fig:
                plt.show()

            plt.close(f)


__all__ = ["BaseState"]

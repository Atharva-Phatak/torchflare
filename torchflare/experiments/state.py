"""Implements Base State."""
import os
from typing import List

import matplotlib.pyplot as plt
import torch

import torchflare.metrics.metric_utils as metric_utils
from torchflare.callbacks.callback import CallbackRunner, sort_callbacks
from torchflare.callbacks.load_checkpoint import LoadCheckpoint
from torchflare.callbacks.model_history import History
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.simple_utils import AvgLoss


class BaseState:
    """Class to set user and internal variables along with some utility functions."""

    def __init__(
        self,
        num_epochs: int,
        save_dir: str,
        model_name: str,
        fp16: bool,
        device: str,
        compute_train_metrics: bool,
        seed: int = 42,
    ):
        """Init method to set up important variables for training and validation.

        Args:
            num_epochs : The number of epochs to save model.
            save_dir : The directory where to save the model, and the log files.
            model_name : The name of '.bin' file.
                Defaults to 'model.bin'
            fp16 : Set this to True if you want to use mixed precision training(Default : False)
            device : The device where you want train your model.
            compute_train_metrics: Whether to compute metrics on training data as well
            seed: The seed to ensure reproducibility.

        Note:
            If batch_mixers are used then set compute_train_metrics to False.
            Also, only validation metrics will be computed if special batch_mixers are used.
        """
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.model_name = model_name
        self.fp16 = fp16
        self.device = device
        self.compute_train_metrics = compute_train_metrics
        self.seed = seed
        self.train_key = "train_"
        self.val_key = "val_"
        self.epoch_key = "Epoch"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.path = os.path.join(self.save_dir, self.model_name)
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        self.history = None
        self.model = None
        self.resume_checkpoint = None
        self.main_metric = None
        self.stop_training = None
        self.experiment_state = None
        self._step_after = None
        self._callback_runner = None
        self.optimizer = None
        self.exp_logs = {}
        self._train_monitor = None
        self._val_monitor = None
        self.criterion = None
        self.compute_metric_flag = True
        self._metric_runner = None
        self.progress_bar = None
        self.header = None
        self._compute_val_metrics = True
        self.train_dl = None
        self.valid_dl = None
        self.loss = None
        self.loss_meter = None
        self.x = None
        self.y = None
        self.preds = None
        self.is_training = None
        self.metrics = None
        self.batch_idx = None
        self.current_epoch = None

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def get_prefix(self):
        """Generates the prefix for training and validation.

        Returns:
            The prefix for training or validation.
        """
        return self.train_key if self.is_training else self.val_key

    def _set_callbacks(self, callbacks: List):
        default_callbacks = [History()]
        if self.resume_checkpoint:
            default_callbacks.append(LoadCheckpoint())
        if callbacks is not None:
            default_callbacks.extend(callbacks)

        default_callbacks = sort_callbacks(default_callbacks)
        self._callback_runner = CallbackRunner(callbacks=default_callbacks)
        self._callback_runner.set_experiment(self)

    def _set_metrics(self, metrics):

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

    @property
    def set_state(self):
        """Returns the current state of experiment.

        Returns:
            The current experiment state.
        """
        return self.experiment_state

    @set_state.setter
    def set_state(self, state: str):

        self.experiment_state = state
        # Run callbacks on state change
        self._callback_runner(current_state=self.experiment_state)

    def _model_to_device(self):
        """Function to move model to device."""
        if next(self.model.parameters()).is_cuda is False:
            self.model.to(self.device)

    def _reset_model_logs(self):
        if bool(self.exp_logs):
            self.exp_logs = {}

    def plot_history(self, keys: List[str], save_fig: bool = False, plot_fig: bool = True):
        """Method to plot model history.

        Args:
            keys: A key value in lower case. Ex accuracy or loss
            save_fig: Set to True if you want to save_fig.
            plot_fig: Whether to plot the figure.
        """
        for key in keys:
            plt.style.use("seaborn")
            plt.figure()
            for k, v in self.history.items():
                if key in k:
                    plt.plot(self.history.get(self.epoch_key), v, "-o", label=k)
            plt.title(f"{key.upper()}/{self.epoch_key.upper()}", fontweight="bold")
            plt.ylabel(f"{key.upper()}", fontweight="bold")
            plt.xlabel(self.epoch_key.upper(), fontweight="bold")
            plt.grid(True)
            plt.legend(loc="upper left")

            if save_fig is not None:
                save_path = os.path.join(self.save_dir, f"{key}-vs-{self.epoch_key.lower()}.jpg")
                plt.savefig(save_path, dpi=150)
            if plot_fig:
                plt.show()


__all__ = ["BaseState"]

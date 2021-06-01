"""Implements Base State."""
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchflare.callbacks.callback import sort_callbacks
from torchflare.callbacks.metric_utils import MetricCallback
from torchflare.callbacks.model_history import History
from torchflare.callbacks.progress_bar import ProgressBar
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.simple_utils import AvgLoss, numpy_to_torch
from torchflare.utils.seeder import seed_all


class Engine:
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
        self.main_metric = None
        self.stop_training = None
        self.stage = None
        self.callbacks = None
        self.optimizer = None
        self.criterion = None
        self._step = None
        self.exp_logs = None
        self.history = None
        self.monitors = {"Train": None, "Valid": None}
        self._metric_runner = None
        self.dataloaders = None
        self.loss, self.loss_meter = None, None
        self.x, self.y = None, None
        self.preds = None
        self.batch_idx, self.current_epoch = None, 0
        self.plot_dir = "./plots"

    def get_prefix(self):
        """Generates the prefix for training and validation.

        Returns:
            The prefix for training or validation.
        """
        return self.train_key if self.stage.startswith("Train") else self.val_key

    def _run_callbacks(self, event):
        for callback in self.callbacks:
            try:
                _ = getattr(callback, event)(self)
            except AttributeError:
                pass

    def get_model_params(self):
        """Create model params for optimizer."""
        grad_params = (param for param in self.model.parameters() if param.requires_grad)
        return grad_params

    def _set_optimizer(self, optimizer, optimizer_params):
        grad_params = self.get_model_params()
        self.optimizer = get_optimizer(optimizer)(grad_params, **optimizer_params)

    def _set_callbacks(self, callbacks: List):
        self.callbacks = [ProgressBar(), History(), AvgLoss()]
        if self._metric_runner is not None:
            self.callbacks.append(self._metric_runner)
        if callbacks is not None:
            self.callbacks.extend(callbacks)

        self.callbacks = sort_callbacks(self.callbacks)

    def _set_metrics(self, metrics):

        if metrics is not None:
            self._metric_runner = MetricCallback(metrics=metrics)

    def _set_model(self, model_class, model_params):
        self.model = model_class(**model_params)

    def _set_criterion(self, criterion):
        self.criterion = get_criterion(criterion)

    def zero_grad(self) -> None:
        """Wrapper for optimizer.zero_grad()."""
        self.optimizer.zero_grad()

    def backward_loss(self) -> None:
        """Method to propogate loss backward."""
        # skipcq: PYL-W0106
        self.scaler.scale(self.loss).backward() if self.fp16 else self.loss.backward()

    def optimizer_step(self) -> None:
        """Method to perform optimizer step."""
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    @staticmethod
    def _dataloader_from_data(args, dataloader_kwargs):
        args = numpy_to_torch(args)
        dataset = TensorDataset(*args) if len(args) > 1 else args[0]
        return DataLoader(dataset, **dataloader_kwargs)

    def _model_to_device(self):
        """Function to move model to device."""
        # skipcq : PTC-W0063
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
        self.exp_logs = {}
        self.stage = None
        self.loss, self.x, self.y, self.preds = None, None, None, None
        self.batch_idx, self.current_epoch = None, None

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

        if save:
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

    def get_logs(self):
        """Returns experiment logs as a dataframe."""
        return pd.DataFrame.from_dict(self.history)


__all__ = ["Engine"]

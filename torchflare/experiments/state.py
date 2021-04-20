"""Implements Experiment State."""
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import torch
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader

import torchflare.batch_mixers as special_augs
import torchflare.metrics.metric_utils as metric_utils
from torchflare.callbacks.callback import CallbackRunner, sort_callbacks
from torchflare.callbacks.load_checkpoint import LoadCheckpoint
from torchflare.callbacks.model_history import History
from torchflare.callbacks.timer import TimeCallback
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.scheduler_utilities import LRScheduler
from torchflare.experiments.simple_utils import wrap_metric_names
from torchflare.utils.seeder import seed_all


class ExperimentState:
    """Class to set user and internal variables along with some utility functions."""

    def __init__(
        self,
        num_epochs: int,
        save_dir: str,
        model_name: str,
        fp16: bool,
        device: str,
        compute_train_metrics: bool,
        using_batch_mixers: bool,
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
            using_batch_mixers : Whether using special batch_mixers like cutmix or mixup.
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
        self._using_batch_mixers = using_batch_mixers
        self.seed = seed
        self.train_key = "train_"
        self.val_key = "val_"
        self.epoch_key = "Epoch"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.path = os.path.join(self.save_dir, self.model_name)
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        self.history = History()
        self.model = None
        self.resume_checkpoint = None
        self.main_metric = None
        self.stop_training = None
        self.experiment_state = None
        self._step_after = None
        self._callback_runner = None
        self.optimizer = None
        self.scheduler_stepper = None
        self.exp_logs = {}
        self._train_monitor = {}
        self._val_monitor = {}
        self.criterion = None
        self.compute_metric_flag = True
        self._metric_runner = None
        self.master_bar = None
        self.progress_bar = None
        self.header = None
        if self.compute_train_metrics is True and self._using_batch_mixers is True:
            warnings.warn(
                "Using special batch_mixers is set to True and compute train metrics is also set to true. "
                "Setting  compute train metrics to False."
            )

            self.compute_train_metrics = False

        self._compute_val_metrics = True
        self._train_dl = None
        self._valid_dl = None

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def _write_stdout(self, stats):
        self.master_bar.write(
            [f"{val:.5f}" if isinstance(val, float) else str(val) for val in stats], table=True,
        )

    def _set_callbacks(self, callbacks: List):
        if callbacks is None:
            if self.resume_checkpoint:
                callbacks = [self.history, TimeCallback(), LoadCheckpoint()]
            else:
                callbacks = [self.history, TimeCallback()]
        else:
            if self.resume_checkpoint:
                callbacks = [self.history, TimeCallback(), LoadCheckpoint()] + callbacks
            else:
                callbacks = [self.history, TimeCallback()] + callbacks

        callbacks = sort_callbacks(callbacks)
        self._callback_runner = CallbackRunner(callbacks=callbacks)
        self._callback_runner.set_experiment(self)

    def _set_metrics(self, metrics):

        self._metric_runner = metric_utils.MetricAndLossContainer(metrics=metrics)
        self._metric_runner.set_experiment(self)
        self.header = self._create_metric_lists(metrics=metrics)

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

    def _set_scheduler(self, scheduler, scheduler_params):
        if scheduler is not None:
            self.scheduler_stepper = LRScheduler(scheduler=scheduler, optimizer=self.optimizer, **scheduler_params)
            self.scheduler_stepper.set_experiment(self)

    def _set_criterion(self, criterion):

        self.criterion = get_criterion(criterion=criterion)
        if self._using_batch_mixers:
            self.criterion = special_augs.MixCriterion(criterion=self.criterion)

    def _create_metric_lists(self, metrics):
        train_l = [self.epoch_key, self.train_key + "loss"]
        val_l = [self.val_key + "loss"]
        if metrics is not None:
            metric_names = wrap_metric_names(metric_list=metrics)
            train_metrics = [(self.train_key + k) for k in metric_names] if self.compute_train_metrics else []
            val_metrics = [(self.val_key + k) for k in metric_names]
            train_l.extend(train_metrics)
            val_l.extend(val_metrics)

        train_l.extend(val_l)
        train_l.append("Time")
        return train_l

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
        epoch = self.exp_logs.pop(self.epoch_key) if self.epoch_key in self.exp_logs.keys() else None
        self._callback_runner(current_state=self.experiment_state, epoch=epoch, logs=self.exp_logs)

    def _model_to_device(self):
        """Function to move model to device."""
        if next(self.model.parameters()).is_cuda is False:
            self.model.to(self.device)

    def _reset_model_logs(self):
        if bool(self.exp_logs):
            self.exp_logs = {}

    def initialize(self, train_dl: DataLoader, valid_dl: DataLoader):
        """Function to move model to device and seed everything.

        Args:
            train_dl : The training dataloader.
            valid_dl : The validation dataloader.
        """
        seed_all(self.seed)
        self._model_to_device()
        self._reset_model_logs()
        self.master_bar = master_bar(range(self.num_epochs))
        self._train_dl = train_dl
        self._valid_dl = valid_dl
        self._write_stdout(stats=self.header)

    def _create_bar(self, iterator):
        """Function to create progress bar.

        Args:
            iterator: The iterator to be used for progress bar.
        """
        self.progress_bar = progress_bar(iterator, parent=self.master_bar, leave=False)
        self.progress_bar.update(0)

    def _update_pbar(self, prefix, val):
        self.progress_bar.comment = f"{prefix + 'loss'} : {str(val)}"

    def _before_step(self, iterator, prefix):
        # before every train/val step create progress bar.
        self._create_bar(iterator=iterator)
        self.compute_metric_flag = self.compute_train_metrics if "train_" in prefix else self._compute_val_metrics

    def cleanup(self):
        """Function to reset the states of monitors and model."""
        self._train_monitor, self._val_monitor, self.exp_logs = {}, {}, {}
        self.experiment_state = None
        self.master_bar = None

    def _run_event(self, event: str, **kwargs):
        _ = getattr(self, event)(**kwargs)

    def plot_history(self, key: str, save_fig: bool = False, plot_fig: bool = True):
        """Method to plot model history.

        Args:
            key : A key value in lower case. Ex accuracy or loss
            save_fig: Set to True if you want to save_fig.
            plot_fig: Whether to plot the figure.
        """
        plt.figure(figsize=(10, 10))
        for k, v in self.history.history.items():
            if key in k:
                plt.plot(v, label=k)
        plt.title(f"{key}/{self.epoch_key}")
        plt.ylabel(f"{key}")
        plt.xlabel(self.epoch_key)
        plt.legend(loc="upper left")

        if save_fig is not None:
            save_path = os.path.join(self.save_dir, f"{key}-vs-{self.epoch_key}.jpg")
            plt.savefig(save_path, dpi=150)
        if plot_fig:
            plt.show()


__all__ = ["ExperimentState"]

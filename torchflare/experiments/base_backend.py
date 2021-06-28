"""Implements Base State."""

import contextlib
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchflare.callbacks.callback import sort_callbacks
from torchflare.callbacks.metric_utils import MetricCallback
from torchflare.callbacks.model_history import History
from torchflare.callbacks.progress_bar import ProgressBar
from torchflare.experiments.core import State
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.simple_utils import AvgLoss, get_name, numpy_to_torch, to_device
from torchflare.utils.seeder import seed_all


# noinspection PyMethodMayBeStatic
class BaseBackend:
    """Class to perform steps for optimizer , scaling etc."""

    def __init__(self):
        self.autocast = contextlib.nullcontext()

    # skipcq :  PYL-R1705
    def zero_grad(self, optimizer) -> None:
        """Wrapper for optimizer.zero_grad()."""
        optimizer.zero_grad()

    # skipcq :  PYL-R1705
    def backward_loss(self, loss) -> None:
        """Method to propogate loss backward."""
        # skipcq: PYL-W0106
        loss.backward()

    # skipcq :  PYL-R1705
    def optimizer_step(self, optimizer) -> None:
        """Method to perform optimizer step."""
        optimizer.step()


# noinspection PyMethodMayBeStatic
class AMPBackend:
    """Class to perform steps for optimizer , scaling using mixed precision."""

    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast = torch.cuda.amp.autocast()

    # skipcq :  PYL-R1705
    def zero_grad(self, optimizer) -> None:
        """Wrapper for optimizer.zero_grad()."""
        optimizer.zero_grad()

    def backward_loss(self, loss) -> None:
        """Method to propogate loss backward."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, optimizer) -> None:
        """Method to perform optimizer step."""
        self.scaler.step(optimizer)
        self.scaler.update()


class BaseExperiment:
    """Base class to handle all the internals for Experiment class."""

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
        self.train_stage, self.valid_stage = "train", "eval"
        self.input_key, self.target_key = "inputs", "targets"
        self.which_loader = None
        self.main_metric = None
        self.stop_training = None
        self._step = None
        self.exp_logs = None
        self.history = None
        self.monitors = {self.train_stage: None, self.valid_stage: None}
        self.loss_per_batch = None
        self.loss = None
        self.batch = None
        self.preds = None
        self.batch_idx, self.current_epoch = None, 0
        self.backend = AMPBackend() if self.fp16 else BaseBackend()
        self.state = None
        self.callback_state = None

    def get_prefix(self):
        """Generates the prefix for training and validation.

        Returns:
            The prefix for training or validation.
        """
        return self.train_key if self.which_loader.startswith(self.train_stage) else self.val_key

    def init_state(self, config, callbacks, metrics):
        """Method to initialise the internal state for experiment.

        Args:
            config: The ModelConfig object.
            callbacks: The input callbacks.
            metrics: The input metrics.
        """
        self.state = State(model=self.init_model(config=config))
        self.state.update(
            {
                "optimizer": self.init_optimizer(config=config),
                "criterion": self.init_criterion(config=config),
                "callbacks": self.init_callbacks(callbacks=callbacks, metrics=metrics),
            }
        )
        if callbacks is not None:
            self.callback_state = State()
            self.callback_state.update({get_name(k): k for k in callbacks})

    def _run_callbacks(self, event):
        for callback in self.state.callbacks:
            try:
                _ = getattr(callback, event)(self)
            except AttributeError:
                pass

    def _process_batch(self, batch):
        if len(batch) == 2:
            batch_dict = {self.input_key: batch[0], self.target_key: batch[1]}
            return to_device(batch_dict, self.device)
        elif isinstance(batch, torch.Tensor) or len(batch) == 1:
            batch_dict = {self.input_key: batch}
            return to_device(batch_dict, self.device)

    @staticmethod
    def get_params(model):
        """Method to get model parameters.

        Args:
            model: The nn.Module object.
        """
        return (param for param in model.parameters() if param.requires_grad)

    def get_model_params(self, optimizer):
        """Create model params for optimizer.

        Args:
            optimizer: The optimizer to be used for Training.
        """
        if isinstance(self.state.model, dict):
            grad_params = {
                opt_key: self.get_params(self.state.model[m_key]) for m_key, opt_key in zip(self.state.model, optimizer)
            }

        else:
            grad_params = self.get_params(self.state.model)

        return grad_params

    def init_optimizer(self, config):
        """Method to initialise the optimizer.

        Args:
            config: The ModelConfig object.
        """
        grad_params = self.get_model_params(config.optimizer)
        if config.optimizer_dict:
            optimizer = {
                k: get_optimizer(o_class)(grad_params[k], **config.optimizer_params[k])
                for k, o_class in config.optimizer.items()
            }
        else:
            optimizer = get_optimizer(config.optimizer)(grad_params, **config.optimizer_params)

        return optimizer

    @staticmethod
    def init_callbacks(callbacks: List, metrics: List):
        """Method to initialise the callbacks.

        Args:
            callbacks: The list of callbacks.
            metrics: The list of metrics.
        """
        cbs = [ProgressBar(), History(), AvgLoss()]
        if metrics is not None:
            cbs.append(MetricCallback(metrics=metrics))
        if callbacks is not None:
            cbs.extend(callbacks)

        cbs = sort_callbacks(cbs)
        return cbs

    @staticmethod
    def init_model(config):
        """Method to initialise the model.

        Args:
            config: The ModelConfig object.
        """
        if config.model_dict:
            models = {k: m_class(**config.module_params[k]) for k, m_class in config.nn_module.items()}
        else:
            models = config.nn_module(**config.module_params)
        return models

    @staticmethod
    def init_criterion(config):
        """Method to initialise the criterion for training.

        Args:
            config: The ModelConfig object.
        """
        if isinstance(config.criterion, dict):
            criterion = {k: get_criterion(crit) for k, crit in config.criterion.items()}
        else:
            criterion = get_criterion(config.criterion)

        return criterion

    @staticmethod
    def _dataloader_from_data(args, dataloader_kwargs):
        args = numpy_to_torch(args)
        dataset = TensorDataset(*args) if len(args) > 1 else args[0]
        return DataLoader(dataset, **dataloader_kwargs)

    @staticmethod
    def _check_model_on_device(model):
        try:
            return next(model.parameters()).is_cuda
        except StopIteration as e:
            print(str(e))

    def _model_to_device(self):
        """Function to move model to device."""
        # skipcq : PTC-W0063
        if isinstance(self.state.model, dict):
            for k in self.state.model:
                if self._check_model_on_device(self.state.model[k]) is False:
                    self.state.model[k].to(self.device)
        else:
            if self._check_model_on_device(self.state.model) is False:
                self.state.model.to(self.device)

    def _set_model_stage(self, stage):
        if isinstance(self.state.model, dict):
            for k in self.state.model:
                _ = getattr(self.state.model[k], stage)()
        else:
            _ = getattr(self.state.model, stage)()

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
        self.exp_logs = None
        self.which_loader = None
        self.batch = None
        self.loss, self.preds = None, None
        self.batch_idx, self.current_epoch = None, None

    def get_logs(self):
        """Returns experiment logs as a dataframe."""
        return pd.DataFrame.from_dict(self.history)


__all__ = ["BaseExperiment"]

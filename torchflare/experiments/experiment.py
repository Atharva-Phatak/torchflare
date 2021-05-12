"""Implements Base class."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchflare.experiments.simple_utils import to_device
from torchflare.experiments.state import BaseState


class Experiment(BaseState):
    """Simple class for handling boilerplate code for training, validation and Inference."""

    def __init__(
        self,
        num_epochs: int,
        fp16: bool = False,
        device: str = "cuda",
        seed: int = 42,
    ):
        """Init method to set up important variables for training and validation.

        Args:
            num_epochs : The number of epochs to save model.
            fp16 : Set this to True if you want to use mixed precision training.
            device : The device where you want train your model.
            seed: The seed to ensure reproducibility.
        """
        super(Experiment, self).__init__(
            num_epochs=num_epochs,
            fp16=fp16,
            device=device,
            seed=seed,
        )

    def compile_experiment(
        self,
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, str, Any],
        optimizer_params: Dict[str, Union[int, float]],
        criterion: Union[Callable[[torch.Tensor], torch.Tensor], str],
        callbacks: List = None,
        metrics: List = None,
        main_metric: Optional[str] = None,
    ):
        """Configures the model for training and validation.

        Args:
            model: The model to be trained.
            optimizer: The optimizer to be used or name of optimizer.
                        If you pass in the name of the optimizer, only optimizers available in pytorch are supported.
            optimizer_params: The parameters to be used for the optimizer.
            criterion: The loss function to optimize or name of the loss function.
                    If you pass in the name of the loss function,
                    only loss functions available in pytorch can be supported.
            callbacks: The list of callbacks to be used.
            metrics: The list of metrics to be used.
            main_metric: The name of main metric to be monitored. Use lower case version.
                        For examples , use 'accuracy' instead of 'Accuracy'.

        Note:
            Supports all the schedulers implemented in pytorch/transformers except SWA.
            Support for custom scheduling will be added soon.
        """
        self.model = model
        self.main_metric = main_metric
        self._set_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)
        self._set_metrics(metrics=metrics)
        self._set_callbacks(callbacks=callbacks)
        self._set_criterion(criterion=criterion)

    def _process_inputs(self, *args):
        args = to_device(args, self.device)
        return args

    def process_inputs(self, x, y=None):
        """Method to move the inputs and targets to the respective device.

        Args:
            x: The input to the model.
            y: The targets. Defaults to None.
        """
        if y is not None:
            x, y = self._process_inputs(x, y)
            self.x, self.y = x, y

        else:
            x = self._process_inputs(x)
            self.x = x[0] if len(x) == 1 else x

    def _update_logs(self):

        # To-do : Better logs updating
        self.exp_logs = {"Epoch": self.current_epoch, **self._train_monitor, **self._val_monitor}

    def _update_monitors(self):
        if self.is_training:
            self._train_monitor = self.metrics
        else:
            self._val_monitor = self.metrics

    def _update_metrics(self):

        metrics = {} if self._metric_runner is None else self._metric_runner.value
        self.metrics = {**self.loss_meter.value, **metrics}
        self._update_monitors()

    def compute_loss(self) -> None:
        """Computes loss given the inputs and targets."""
        if isinstance(self.preds, (list, tuple)):
            vals = [self.criterion(ele, self.y) for ele in self.preds]
            self.loss = sum(vals)
        else:
            self.loss = self.criterion(self.preds, self.y)

    def model_forward_pass(self):
        """Forward pass of the model."""
        if isinstance(self.x, (list, tuple)):
            self.preds = self.model(*self.x)
        elif isinstance(self.x, dict):
            self.preds = self.model(**self.x)
        else:
            self.preds = self.model(self.x)

    def _calculate_loss(self) -> None:
        """Function to calculate loss and update metric states."""
        self.model_forward_pass()
        self.compute_loss()

    def set_dataloaders(self, train_dl, valid_dl):
        """Setup dataloader variables."""
        self.train_dl = train_dl
        self.valid_dl = valid_dl

    def on_experiment_start(self):
        """Event on experiment start."""
        self.initialise()

    def on_batch_start(self):
        """Event on batch start."""
        self.process_inputs(self.x, self.y)

    def on_loader_start(self):
        """Event on loader start."""
        self._step = self.train_step if self.is_training else self.val_step
        self._iterator = self.train_dl if self.is_training else self.valid_dl

    def on_epoch_start(self):
        """Event on epoch start."""
        self.current_epoch += 1

    def on_experiment_end(self):
        """Event on experiment end."""
        self.cleanup()

    def on_batch_end(self):
        """Event on batch end."""
        self.loss_meter.accumulate()
        # accumulate values for metric computation
        if self._metric_runner is not None:
            self._metric_runner.accumulate()

    def on_loader_end(self):
        """Event of loader end."""
        self._update_metrics()

    def on_epoch_end(self):
        """Event on epoch end."""
        self._update_logs()

    def _run_event(self, event: str):
        """Method to run events."""
        getattr(self, event)()
        # As soon as event ends, we run callbacks.
        self._callback_runner(current_state=event)

    def run_batch(self) -> None:
        """Run batch with batch event."""
        self._run_event("on_batch_start")
        self._step()
        self._run_event("on_batch_end")

    def run_loader(self):
        """Function to iterate the dataloader through all the batches."""
        self._run_event("on_loader_start")
        for self.batch_idx, (self.x, self.y) in enumerate(self._iterator):
            self.run_batch()
        # Stdout  computed metrics/update monitors.
        self._run_event("on_loader_end")

    def fp16_step(self) -> None:
        """Method to perform mixed precision type update."""
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            self._calculate_loss()

        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def standard_step(self) -> None:
        """Method to perform the standard update."""
        self.optimizer.zero_grad()
        self._calculate_loss()
        self.loss.backward()
        self.optimizer.step()

    def train_step(self) -> None:
        """Method to perform the train step and step scheduler."""
        # skipcq: PYL-W0106
        self.fp16_step() if self.fp16 else self.standard_step()

    def val_step(self) -> None:
        """Method to perform validation step."""
        with torch.no_grad():
            self._calculate_loss()

    def _do_train_epoch(self):
        """Method to train the model for one epoch."""
        self.model.train()
        self.is_training = True
        self.run_loader()

    def _do_val_epoch(self):
        """Method to validate model for one epoch."""
        self.model.eval()
        self.is_training = False
        self.run_loader()

    def _do_epoch(self):
        self._do_train_epoch()
        self._do_val_epoch()

    def _run(self):
        """Method to run experiment for full number of epochs."""
        for _ in range(self.num_epochs):

            self._run_event("on_epoch_start")
            self._do_epoch()
            self._run_event("on_epoch_end")
            if self.stop_training:
                break

    def fit(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        val_data: Union[Tuple, List],
        batch_size: int = 64,
        dataloader_kwargs: Dict = None,
    ):
        """Train and validate the model on training and validation dataset.

        Args:
            x: A numpy array(or array-like) or torch.tensor for inputs to the model.
            y: Target data. Same type as input data coule numpy array(or array-like) or torch.tensors.
            val_data: A tuple or list (x_val , y_val) of numpy arrays or torch.tensors.
            batch_size: The batch size to be used for training and validation.
            dataloader_kwargs: Keyword arguments to pass to the PyTorch dataloaders created
                internally. By default, shuffle=True is passed for the training dataloader but this can be
                overriden by using this argument.

        Note:
            Model will only be saved when ModelCheckpoint callback is used.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        dataloader_kwargs = {"batch_size": batch_size, **dataloader_kwargs}
        train_dl = self._dataloader_from_data((x, y), {"shuffle": True, **dataloader_kwargs})
        valid_dl = self._dataloader_from_data(val_data, dataloader_kwargs)
        self.fit_loader(train_dl=train_dl, valid_dl=valid_dl)

    def fit_loader(self, train_dl: DataLoader, valid_dl: DataLoader):
        """Train and validate the model using dataloaders.

        Args:
            train_dl : The training dataloader.
            valid_dl : The validation dataloader.

        Note:
            Model will only be saved when ModelCheckpoint callback is used.
        """
        self.set_dataloaders(train_dl=train_dl, valid_dl=valid_dl)
        self._run_event("on_experiment_start")
        self._run()
        self._run_event("on_experiment_end")

    @torch.no_grad()
    def _infer_on_batch(self, inp):

        self.process_inputs(x=inp, y=None)
        self.model_forward_pass()

        return self.preds.detach().cpu()

    @torch.no_grad()
    def predict_on_loader(
        self,
        path_to_model: str,
        test_dl: DataLoader,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Method to perform inference on test dataloader.

        Args:
            test_dl: The dataloader to be use for testing.
            device: The device on which you want to perform inference.
            path_to_model: The full path to model

        Yields:
            Output per batch.
        """
        # move model to device
        self._model_to_device()
        ckpt = torch.load(path_to_model, map_location=torch.device(device))
        if isinstance(ckpt, dict):
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        for inp in test_dl:
            op = self._infer_on_batch(inp=inp)
            yield op

    @torch.no_grad()
    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray],
        path_to_model: str,
        batch_size: int = 64,
        dataloader_kwargs: Dict = None,
        device: str = "cuda",
    ):
        """Method to perform inference on test data.

        Args:
            x: A numpy array(or array-like) or torch.tensor for inputs to the model.
            batch_size: The batch size to be used for inference.
            device: The device on which you want to perform inference.
            dataloader_kwargs: Keyword arguments to pass to the PyTorch dataloader which is created internally.
            path_to_model: str,
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        dataloader_kwargs = {"batch_size": batch_size, **dataloader_kwargs}
        dl = self._dataloader_from_data((x,), dataloader_kwargs)
        return self.predict_on_loader(path_to_model=path_to_model, test_dl=dl, device=device)

"""Implements Base class."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchflare.experiments.engine import Engine
from torchflare.experiments.simple_utils import _has_intersection, to_device


class Experiment(Engine):
    """Simple class for handling boilerplate code for training, validation and Inference.

    Args:
           num_epochs(int) : The number of epochs to save model.
           fp16(bool) : Set this to True if you want to use mixed precision training.
           device(str) : The device where you want train your model. One of **cuda** or **cpu**.
           seed(int): The seed to ensure reproducibility.

    Examples:
        .. code-block:: python

            import torch
            import torchflare.callbacks as cbs
            import torchflare.metrics as metrics
            from torchflare.experiments import Experiment

            # Defining Training/Validation Dataloaders
            train_dl = SomeTrainDataloader()
            valid_dl = SomeValidDataloader()

            # Defining params
            optimizer = "Adam"
            optimizer_params = {"lr" : 1e-4}
            criterion = "cross_entropy"
            num_epochs = 10
            num_classes = 4

            # Defining the list of metrics
            metric_list = [
                metrics.Accuracy(num_classes=num_classes, multilabel=False),
                metrics.F1Score(num_classes=num_classes, multilabel=False),
            ]

            # Defining the list of callbacks
            callbacks = [
                cbs.EarlyStopping(monitor="accuracy", mode="max"),
                cbs.ModelCheckpoint(monitor="accuracy", mode = "max"),
                cbs.ReduceLROnPlateau(mode = "max" , patience = 3) #Defining Scheduler callback.
            ]

            # Creating Experiment and setting the params.
            exp = Experiment(
                num_epochs=num_epochs,
                fp16=True,
                device=device,
                seed=42,
            )

            # Compiling the experiment
            exp.compile_experiment(
                module=SomeModelClass,
                module_params = {"num_features" : 200 , "num_classes" : 5} #Params to init the model class
                metrics=metric_list,
                callbacks=callbacks,
                main_metric="accuracy",
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                criterion=criterion,
            )


            # Running the experiment
            exp.fit_loader(train_dl=train_dl, valid_dl=valid_dl)
    """

    def __init__(
        self,
        num_epochs: int,
        fp16: bool = False,
        device: str = "cuda",
        seed: int = 42,
    ):
        """Init method to set up important variables for training and validation."""
        super(Experiment, self).__init__(
            num_epochs=num_epochs,
            fp16=fp16,
            device=device,
            seed=seed,
        )

    def compile_experiment(
        self,
        module: nn.Module,
        module_params: Optional[Dict],
        optimizer: Union[torch.optim.Optimizer, str, Any],
        optimizer_params: Optional[Dict],
        criterion: Union[Callable, str],
        callbacks: List = None,
        metrics: List = None,
        main_metric: Optional[str] = None,
    ):
        """Configures the model for training and validation.

        Args:
            module(nn.Module): An uninstantiated PyTorch class which defines the model.
            module_params(Dict): The params required to initialize model class.
            optimizer(torch.optim.Optimizer, str): The optimizer to be used or name of optimizer.
                        If you pass in the name of the optimizer, only optimizers available in pytorch are supported.
            optimizer_params(Dict): The parameters for optimizer.
            criterion(callable , str): The loss function to optimize or name of the loss function.
                    If you pass in the name of the loss function,
                    only loss functions available in pytorch can be supported.
            callbacks(List): The list of callbacks to be used.
            metrics(List): The list of metrics to be used.
            main_metric(str): The name of main metric to be monitored. Use lower case version.
                        For examples , use 'accuracy' instead of 'Accuracy'.

        Note:
            Supports all the schedulers implemented in pytorch/transformers except SWA.
            Support for custom scheduling will be added soon.
        """
        self.main_metric = main_metric
        self._set_metrics(metrics=metrics)
        self._set_model(model_class=module, model_params=module_params)
        self._set_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)
        self._set_metrics(metrics=metrics)
        self._set_criterion(criterion=criterion)
        self._set_callbacks(callbacks=callbacks)

    def _process_inputs(self, *args):
        args = to_device(args, self.device)
        return args

    # noinspection DuplicatedCode
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

    def compute_loss(self) -> None:
        """Computes loss given the inputs and targets."""
        self.loss = self.criterion(self.preds, self.y)

    def model_forward_pass(self):
        """Forward pass of the model."""
        if isinstance(self.x, (list, tuple)):
            self.preds = self.model(*self.x)
        elif isinstance(self.x, dict):
            self.preds = self.model(**self.x)
        else:
            self.preds = self.model(self.x)

    def _handle_batch(self) -> None:
        """Function to calculate loss and update metric states."""
        self.model_forward_pass()
        if self.fp16:
            with torch.cuda.amp.autocast():
                self.compute_loss()
        else:
            self.compute_loss()

    def set_dataloaders(self, train_dl, valid_dl):
        """Setup dataloader variables."""
        self.dataloaders = {"Train": train_dl, "Valid": valid_dl}

    def on_experiment_start(self):
        """Event on experiment start."""
        self.initialise()

    def on_batch_start(self):
        """Event on batch start."""
        self.process_inputs(self.x, self.y)

    def on_loader_start(self):
        """Event on loader start."""
        self._step = {"Train": self.train_step, "Valid": self.val_step}

    def on_epoch_start(self):
        """Event on epoch start."""
        self.current_epoch += 1
        self.exp_logs = {self.epoch_key: self.current_epoch}

    def on_experiment_end(self):
        """Event on experiment end."""
        self.cleanup()

    def on_batch_end(self):
        """Event on batch end."""
        pass

    def on_loader_end(self):
        """Event of loader end."""
        self.exp_logs.update(**self.monitors[self.stage])

    def on_epoch_end(self):
        """Event on epoch end."""
        pass

    def _run_event(self, event: str):
        """Method to run events."""
        if _has_intersection(key="_start", event=event):
            getattr(self, event)()
        # As soon as event ends, we run callbacks.
        self._run_callbacks(event=event)
        if _has_intersection(key="_end", event=event):
            getattr(self, event)()

    def run_batch(self) -> None:
        """Run batch with batch event."""
        self._run_event("on_batch_start")
        self._step.get(self.stage)()
        self._run_event("on_batch_end")

    def run_loader(self):
        """Function to iterate the dataloader through all the batches."""
        self._run_event("on_loader_start")
        iterator = self.dataloaders.get(self.stage)
        for self.batch_idx, (self.x, self.y) in enumerate(iterator):
            self.run_batch()
        # Stdout  computed metrics/update monitors.
        self._run_event("on_loader_end")

    def train_step(self):
        """Method to perform train step."""
        self.zero_grad()
        self._handle_batch()
        self.backward_loss()
        self.optimizer_step()

    def val_step(self) -> None:
        """Method to perform validation step."""
        with torch.no_grad():
            self._handle_batch()

    def _do_train_epoch(self):
        """Method to train the model for one epoch."""
        self.model.train()
        self.stage = "Train"
        self.run_loader()

    def _do_val_epoch(self):
        """Method to validate model for one epoch."""
        self.model.eval()
        self.stage = "Valid"
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

    def _general_fit(self):
        self._run_event("on_experiment_start")
        self._run()
        self._run_event("on_experiment_end")

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
            x(numpy array or torch.Tensor): A numpy array(or array-like) or torch.tensor for inputs to the model.
            y(numpy array or torch.Tensor): Target data. Same type as input data coule numpy array(or array-like)
                                        or torch.tensors.
            val_data(tuple of 2 torch.Tensors or numpy arrays: (input, target): A tuple or list (x_val , y_val) of
                                                                numpy arrays or torch.tensors.
            batch_size(int): The batch size to be used for training and validation.
            dataloader_kwargs(Dict): Keyword arguments to pass to the PyTorch dataloaders created
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
            train_dl(DataLoader) : The training dataloader.
            valid_dl(DataLoader) : The validation dataloader.

        Note:
            Model will only be saved when ModelCheckpoint callback is used.
        """
        self.set_dataloaders(train_dl=train_dl, valid_dl=valid_dl)
        self._general_fit()

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
            test_dl(DataLoader): The dataloader to be use for testing.
            device(str): The device on which you want to perform inference.
            path_to_model(str): The full path to model

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
            x(numpy array or torch.Tensor): A numpy array(or array-like) or torch.tensor for inputs to the model.
            batch_size(int): The batch size to be used for inference.
            device(str): The device on which you want to perform inference.
            dataloader_kwargs(Dict): Keyword arguments to pass to the PyTorch dataloader which is created internally.
            path_to_model(str): The full path to the model.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        dataloader_kwargs = {"batch_size": batch_size, **dataloader_kwargs}
        dl = self._dataloader_from_data((x,), dataloader_kwargs)
        return self.predict_on_loader(path_to_model=path_to_model, test_dl=dl, device=device)


__all__ = ["Experiment"]

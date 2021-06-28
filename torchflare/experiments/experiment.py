"""Implements Base class."""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchflare.experiments.base_backend import BaseExperiment
from torchflare.experiments.simple_utils import _has_intersection, to_device

if TYPE_CHECKING:
    from torchflare.experiments.config import ModelConfig


class Experiment(BaseExperiment):
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
            # Defining the model config which contains model, optimizer, criterion.
            config = ModelConfig(nn_module = SomeModelClass,
                                module_params = {"num_features" : 200 , "num_classes" : 5},
                                optimizer=optimizer,
                                optimizer_params=optimizer_params,
                                criterion = criterion)
            # Compiling the experiment
            exp.compile_experiment(
                model_config = config,
                metrics=metric_list,
                callbacks=callbacks,
                main_metric="accuracy",

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
        model_config: "ModelConfig",
        callbacks: List = None,
        metrics: List = None,
        main_metric: Optional[str] = None,
    ):
        """Configures the model for training and validation.

        Args:
            model_config: An ModelConfig object which holds information about models,
                        optimizer and criterion.
            callbacks(List): The list of callbacks to be used.
            metrics(List): The list of metrics to be used.
            main_metric(str): The name of main metric to be monitored. Use lower case version.
                        For examples , use 'accuracy' instead of 'Accuracy'.

        Note:
            Supports all the schedulers implemented in pytorch/transformers except SWA.
            Support for custom scheduling will be added soon.
        """
        self.main_metric = main_metric
        self.init_state(config=model_config, callbacks=callbacks, metrics=metrics)

    def batch_step(self) -> None:
        """Function to do model forward pass and compute loss."""
        # Note if you overide the batch_step you have to define self.preds, self.loss
        # It is mandatory to define self.loss_per_batch.
        self.preds = self.state.model(self.batch[self.input_key])
        self.loss = self.state.criterion(self.preds, self.batch[self.target_key])
        self.loss_per_batch = {"loss": self.loss.item()}

    def set_dataloaders(self, train_dl, valid_dl):
        """Setup dataloader variables."""
        dataloaders = {self.train_stage: train_dl}
        if valid_dl is not None:
            dataloaders[self.valid_stage] = valid_dl
        self.state.update({"dataloaders": dataloaders})

    def on_experiment_start(self):
        """Event on experiment start."""
        self.initialise()

    def on_batch_start(self):
        """Event on batch start."""
        self.batch = self._process_batch(self.batch)

    def on_loader_start(self):
        """Event on loader start."""
        self._step = {self.train_stage: self.train_step, self.valid_stage: self.val_step}

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
        self.exp_logs.update(**self.monitors[self.which_loader])

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
        self._step.get(self.which_loader)()
        self._run_event("on_batch_end")

    def run_loader(self, dataloader):
        """Function to iterate the dataloader through all the batches."""
        self._run_event("on_loader_start")
        mode = True if self.train_stage in self.which_loader else False
        with torch.set_grad_enabled(mode=mode):
            for self.batch_idx, self.batch in enumerate(dataloader):
                with self.backend.autocast:
                    self.run_batch()
        self._run_event("on_loader_end")

    def train_step(self):
        """Method to perform train step.

        Override this method to perform a custom Training step.
        """
        self.backend.zero_grad(optimizer=self.state.optimizer)
        self.batch_step()
        assert self.loss_per_batch is not None, "Please define self.loss_per_batch for progress bar."
        self.backend.backward_loss(loss=self.loss)
        self.backend.optimizer_step(optimizer=self.state.optimizer)

    def val_step(self) -> None:
        """Method to perform validation step.

        Override this method to perform custom validation step.
        """
        self.batch_step()

    def _do_epoch(self):
        for self.which_loader, dataloader in self.state.dataloaders.items():
            self._set_model_stage(stage=self.which_loader)
            self.run_loader(dataloader=dataloader)

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
        val_data: Optional[Union[Tuple, List]] = None,
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
        if val_data is not None:
            valid_dl = self._dataloader_from_data(val_data, dataloader_kwargs)
        else:
            valid_dl = None
        self.fit_loader(train_dl=train_dl, valid_dl=valid_dl)

    def fit_loader(self, train_dl: DataLoader, valid_dl: DataLoader = None):
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
            self.state.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.state.model.load_state_dict(ckpt)

        for inp in test_dl:
            inp = to_device(inp, device=device)
            op = self.state.model(inp)
            yield op.detach().cpu()

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

"""Implements Base class."""
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchflare.callbacks.states import ExperimentStates
from torchflare.experiments.simple_utils import to_device
from torchflare.experiments.state import BaseState
from torchflare.utils.progress_bar import ProgressBar
from torchflare.utils.seeder import seed_all


class Experiment(BaseState):
    """Simple class for handling boilerplate code for training, validation and Inference."""

    def __init__(
        self,
        num_epochs: int,
        save_dir: str = "./exp_outputs",
        model_name: str = "model.bin",
        fp16: bool = False,
        device: str = "cuda",
        compute_train_metrics: bool = False,
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
            Also, only validation metrics will be computed if  batch_mixers are used.
        """
        super(Experiment, self).__init__(
            num_epochs=num_epochs,
            save_dir=save_dir,
            model_name=model_name,
            fp16=fp16,
            device=device,
            compute_train_metrics=compute_train_metrics,
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
        resume_checkpoint: bool = False,
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
            resume_checkpoint: Whether to resume training from the saved model.

        Note:
            Supports all the schedulers implemented in pytorch/transformers except SWA.
            Support for custom scheduling will be added soon.
        """
        self.model = model
        self.resume_checkpoint = resume_checkpoint
        self.main_metric = main_metric
        seed_all(self.seed)
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

    def _update_model_logs(self):

        # To-do : Better logs updating
        self.exp_logs.update({"Epoch": self.current_epoch, **self._train_monitor, **self._val_monitor})

    # def _step_scheduler(self):
    # if self.scheduler_stepper is not None:
    # self.scheduler_stepper.step(current_state=self.experiment_state)

    def _update_monitors(self, metrics):
        if self.is_training:
            self._train_monitor = metrics
        else:
            self._val_monitor = metrics

    def _create_metric_dict(self):
        if self._metric_runner.metrics is not None:
            self.metrics = self._metric_runner.value
        else:
            self.metrics = {}

        metrics = {**self.loss_meter.value, **self.metrics}
        return metrics

    def _compute_loss(self) -> None:
        """Computes loss given the inputs and targets."""
        if isinstance(self.preds, (list, tuple)):
            vals = [self.criterion(ele, self.y) for ele in self.preds]
            self.loss = sum(vals)
        else:
            self.loss = self.criterion(self.preds, self.y)

    def _model_forward_pass(self):

        if isinstance(self.x, (list, tuple)):
            self.preds = self.model(*self.x)
        elif isinstance(self.x, dict):
            self.preds = self.model(**self.x)
        else:
            self.preds = self.model(self.x)

    def _calculate_loss(self) -> None:
        """Function to calculate loss and update metric states."""
        self.process_inputs(self.x, self.y)
        self._model_forward_pass()
        self._compute_loss()

    def _update_pbar(self):
        self.progress_bar.update(current_step=self.batch_idx, values={"loss": self.loss.item()})

    def set_dataloaders(self, train_dl, valid_dl):
        """Setup dataloader variables."""
        self.train_dl = train_dl
        self.valid_dl = valid_dl

    def on_experiment_start(self):
        """Event on experiment start."""
        self._model_to_device()
        self._reset_model_logs()
        self.progress_bar = ProgressBar(num_epochs=self.num_epochs, train_dl=self.train_dl, valid_dl=self.valid_dl)
        self.set_state = ExperimentStates.EXP_START

    def on_experiment_end(self):
        """Event on experiment end."""
        self.set_state = ExperimentStates.EXP_END
        self._train_monitor, self._val_monitor, self.exp_logs = None, None, {}
        self.experiment_state = None
        self.progress_bar = None
        self.loss, self.x, self.y, self.preds = None, None, None, None

    def on_batch_start(self):
        """Event on batch start."""
        self.set_state = ExperimentStates.BATCH_START

    def on_batch_end(self):
        """Event on batch end."""
        self.loss_meter.accumulate()
        # accumulate values for metric computation
        if self._metric_runner.metrics is not None:
            self._metric_runner.accumulate()
        self._update_pbar()  # per batch loss.
        self.set_state = ExperimentStates.BATCH_END

    def on_loader_start(self):
        """Event on batch end."""
        # before every train/val step create progress bar.
        self.set_state = ExperimentStates.LOADER_START
        self.progress_bar.set_steps(is_training=self.is_training)
        if self._metric_runner.metrics is not None:
            self.compute_metric_flag = self.compute_train_metrics if self.is_training else self._compute_val_metrics

    def on_loader_end(self):
        """Method to print final metrics and reset state of progress bar."""
        # compute metrics
        self.set_state = ExperimentStates.LOADER_END
        metrics = self._create_metric_dict()
        self.progress_bar.add(n=1, values=metrics)
        self.progress_bar.reset()
        self._update_monitors(metrics=metrics)

    def on_epoch_start(self):
        """Event on epoch start."""
        self.progress_bar.display_epoch(current_epoch=self.current_epoch)
        self.set_state = ExperimentStates.EPOCH_START

    def on_epoch_end(self):
        """Event on epoch end."""
        self._update_model_logs()
        # self._step_scheduler()
        # RUNNING CALLBACKS AFTER EVERY EPOCH IS DONE
        self.set_state = ExperimentStates.EPOCH_END

    def _run_event(self, event: str):
        """Method to run events."""
        getattr(self, event)()

    def run_loader(self, func: Callable):
        """Function to iterate the dataloader through all the batches.

        Args:
            func: The function which will compute the loss , outputs and return them.
        """
        # create progress bar and set compute_flag
        self._run_event("on_loader_start")
        iterator = self.train_dl if self.is_training else self.valid_dl

        for self.batch_idx, (self.x, self.y) in enumerate(iterator):
            self._run_event("on_batch_start")
            func()
            self._run_event("on_batch_end")

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
        if self.fp16:
            self.fp16_step()
        else:
            self.standard_step()
        # self._step_scheduler()

    def val_step(self) -> None:
        """Method to perform validation step."""
        with torch.no_grad():
            self._calculate_loss()

    def _do_train_epoch(self):
        """Method to train the model for one epoch."""
        self.model.train()
        self.is_training = True
        self.run_loader(func=self.train_step)

    def _do_val_epoch(self):
        """Method to validate model for one epoch."""
        self.model.eval()
        self.is_training = False
        self.run_loader(func=self.val_step)

    def _do_epoch(self):
        self._do_train_epoch()
        self._do_val_epoch()

    def _run(self):
        """Method to run experiment for full number of epochs."""
        for self.current_epoch in range(self.num_epochs):

            self._run_event("on_epoch_start")
            self._do_epoch()
            self._run_event("on_epoch_end")
            if self.stop_training:
                break

    def run_experiment(self, train_dl: DataLoader, valid_dl: DataLoader):
        """Method to train and validate the model for fixed number of epochs.

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
        self._model_forward_pass()

        return self.preds.detach().cpu()

    @torch.no_grad()
    def infer(self, test_dl: torch.utils.data.DataLoader, path: str, device: str = "cuda") -> torch.Tensor:
        """Method to perform inference on test dataloader.

        Args:
            test_dl: The dataloader to be use for testing.
            device: The device on which you want to perform inference.
            path: The path where the '.bin' or '.pt' or '.pth' file is saved.
                example: path = "/output/model.bin"

        Yields:
            Output per batch
        """
        # move model to device
        self._model_to_device()
        ckpt = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(ckpt["model_state_dict"])

        for inp in test_dl:
            op = self._infer_on_batch(inp=inp)
            yield op

    # Perform a sanity check for the forward pass of them model. Ensuring model is defined correctly.
    def perform_sanity_check(self, dl: DataLoader):
        """Method to check if the model forward pass and loss_computation is working or not.

        Args:
            dl: A PyTorch dataloader.
        """
        self._model_to_device()
        self.x, self.y = next(iter(dl))
        self._calculate_loss()
        print("Sanity Check Completed. Model Forward Pass and Loss Computation Successful")
        print(f"Output Shape : {self.preds.shape}")
        print(f"Loss for a batch :{self.loss.item()}")

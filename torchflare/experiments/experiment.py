"""Implements Experiment class."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchflare.callbacks.states import ExperimentStates
from torchflare.experiments.simple_utils import to_device
from torchflare.experiments.state import ExperimentState


class Experiment(ExperimentState):
    """Simple Experiment for handling boilerplate code for training, validation and Inference."""

    def __init__(
        self,
        num_epochs: int,
        save_dir: str = "./exp_outputs",
        model_name: str = "model.bin",
        fp16: bool = False,
        device: str = "cuda",
        compute_train_metrics: bool = False,
        using_batch_mixers: bool = False,
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
            Also, only validation metrics will be computed if  batch_mixers are used.
        """
        super(Experiment, self).__init__(
            num_epochs=num_epochs,
            save_dir=save_dir,
            model_name=model_name,
            fp16=fp16,
            device=device,
            compute_train_metrics=compute_train_metrics,
            using_batch_mixers=using_batch_mixers,
            seed=seed,
        )

    def compile_experiment(
        self,
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, str, Any],
        optimizer_params: Dict[str, Union[int, float]],
        criterion: Union[Callable[[torch.Tensor], torch.Tensor], str],
        scheduler: Optional[str] = None,
        scheduler_params: Dict[str, Union[int, float, str]] = None,
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
            scheduler: The scheduler or the name of the scheduler.
                    If you pass in the name of the scheduler, only scheduler available in
                    pytorch/transformers can be supported.
            scheduler_params: The parameters to be used for the scheduler.
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

        self._set_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)
        self._set_scheduler(scheduler=scheduler, scheduler_params=scheduler_params)
        self._set_metrics(metrics=metrics)
        self._set_callbacks(callbacks=callbacks)
        self._set_criterion(criterion=criterion)

    @property
    def update_train_monitor(self):
        """Returns the train monitor dictionary.

        Returns:
            Training monitor dictionary.
        """
        return self._train_monitor

    @update_train_monitor.setter
    def update_train_monitor(self, metrics: Dict):
        self._train_monitor.update(metrics)

    @property
    def update_val_monitor(self):
        """Returns the validation  monitor dictionary.

        Returns:
            Validation monitor dictionary.
        """
        return self._val_monitor

    @update_val_monitor.setter
    def update_val_monitor(self, metrics: Dict):
        self._val_monitor.update(metrics)

    def _update_mbar(self, epoch):
        logs = [epoch] + list(self.exp_logs.values())
        self._write_stdout(stats=logs)

    def _process_inputs(self, *args):
        args = to_device(args, self.device)
        return args

    def process_inputs(self, x, y=None):
        """Method to move the inputs and targets to the respective device.

        Args:
            x: The input to the model.
            y: The targets. Defaults to None.

        Returns:
            The input and targets if targets are present else returns inputs
        """
        if y is not None:
            x, y = self._process_inputs(x, y)
            return x, y

        else:
            x = self._process_inputs(x)
            return x[0] if len(x) == 1 else x

    def _update_model_logs(self, epoch: int = None):

        # To-do : Better logs updating
        self.exp_logs.update({"Epoch": epoch, **self._train_monitor, **self._val_monitor})

    def _compute_loss(self, op: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Computes loss given the inputs and targets.

        Args:
            op: The output of the net
            y: The targets

        Returns:
            Computed loss
        """
        if isinstance(op, (list, tuple)):
            vals = [self.criterion(ele, y) for ele in op]
            loss = sum(vals)
        else:
            loss = self.criterion(op, y)

        return loss

    def _model_forward_pass(self, x):

        if isinstance(x, (list, tuple)):
            op = self.model(*x)
        elif isinstance(x, dict):
            op = self.model(**x)
        else:
            op = self.model(x)

        return op

    def _calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to calculate loss and update metric states.

        Args:
            x : The input to the network
            y: The targets.

        Returns:
              The computed loss and output
        """
        x, y = self.process_inputs(x, y)
        op = self._model_forward_pass(x=x)
        loss = self._compute_loss(op=op, y=y)

        return loss, op

    def run_batches(self, iterator, prefix: str, func: Callable):
        """Function to iterate the dataloader through all the batches.

        Args:
            iterator: The dataloader or any iterator which will yield the inputs and targets.
            prefix: The prefix for training/validation. .
            func: The function which will compute the loss , outputs and return them.

        Returns:
            A dictionary containing metrics and loss.
        """
        # create progress bar and set compute_flag
        self._before_step(iterator=iterator, prefix=prefix)

        # reset metrics
        self._metric_runner.reset()

        for x, y in self.progress_bar:
            self.set_state = ExperimentStates.BATCH_START

            loss, op = func(x, y)

            # accumulate values for metric computation
            self._metric_runner.accumulate(op=op, y=y, loss=loss.item(), n=iterator.batch_size)
            self._update_pbar(prefix=prefix, val=loss.item())

            self.set_state = ExperimentStates.BATCH_END

        # compute metrics
        metrics = self._metric_runner.compute(prefix=prefix)
        return metrics

    def fp16_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to perform mixed precision type update.

        Args:
            inputs : The inputs obtained from the dataloader
            targets: The targets obtained from the dataloader

        Returns:
            The computed loss and outputs
        """
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, op = self._calculate_loss(x=inputs, y=targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, op

    def standard_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to perform the standard update.

        Args:
            inputs : The inputs obtained from the dataloader
            targets: The targets obtained from the dataloader

        Returns:
            The computed loss and outputs
        """
        self.optimizer.zero_grad()
        loss, op = self._calculate_loss(x=inputs, y=targets)
        loss.backward()
        self.optimizer.step()

        return loss, op

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to perform the train step and step scheduler.

        Args:
            inputs: The input to the model.
            targets: The targets.

        Returns:
            Return loss and model output.
        """
        func = "fp16_step" if self.fp16 else "standard_step"
        loss, op = getattr(self, func)(inputs=inputs, targets=targets)
        if self.scheduler_stepper is not None:
            self.scheduler_stepper.step(current_state=self.experiment_state)
        return loss, op

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to perform validation step.

        Args:
            inputs: The input to the model.
            targets: The targets.

        Returns:
            Return loss and model outputs.
        """
        with torch.no_grad():
            loss, op = self._calculate_loss(x=inputs, y=targets)
        return loss, op

    def _do_train_epoch(self):
        """Method to train the model for one epoch."""
        self.model.train()
        metrics = self.run_batches(iterator=self._train_dl, prefix=self.train_key, func=self.train_step)

        # GENERATE AND UPDATE METRIC MONITORS
        self.update_train_monitor = metrics

    def _do_val_epoch(self):
        """Method to validate model for one epoch."""
        self.model.eval()
        metrics = self.run_batches(iterator=self._valid_dl, prefix=self.val_key, func=self.val_step)

        # GENERATE AND UPDATE THE MONITORS
        self.update_val_monitor = metrics

    def _do_epoch(self, current_epoch):
        self._do_train_epoch()
        self._do_val_epoch()
        self._update_model_logs(epoch=current_epoch)
        if self.scheduler_stepper is not None:
            self.scheduler_stepper.step(current_state=self.experiment_state)

    def _run(self):
        """Method to run experiment for full number of epochs."""
        for epoch in self.master_bar:

            self.set_state = ExperimentStates.EPOCH_START
            self._do_epoch(current_epoch=epoch)

            # RUNNING CALLBACKS AFTER EVERY EPOCH IS DONE
            self.set_state = ExperimentStates.EPOCH_END
            self._update_mbar(epoch=epoch)

            if self.stop_training:
                break

    def run_experiment(
        self, train_dl: DataLoader, valid_dl: DataLoader,
    ):
        """Method to train and validate the model for fixed number of epochs.

        Args:
            train_dl : The training dataloader.
            valid_dl : The validation dataloader.

        Note:
            Model will only be saved when ModelCheckpoint callback is used.
        """
        # Seed and move to model to device
        self._run_event(event="initialize", train_dl=train_dl, valid_dl=valid_dl)
        # Set the state to TRAIN START
        self.set_state = ExperimentStates.EXP_START
        self._run()
        # Set the state to TRAIN END and run callbacks to ensure everything is completed
        self.set_state = ExperimentStates.EXP_END
        # Perform cleanup , set monitors and logs to empty dicts
        self._run_event(event="cleanup")

    @torch.no_grad()
    def _infer_on_batch(self, inp):

        inp = self.process_inputs(x=inp, y=None)
        op = self._model_forward_pass(x=inp)

        return op.detach().cpu()

    @torch.no_grad()
    def infer(self, test_loader: torch.utils.data.DataLoader, path: str, device: str = "cuda") -> torch.Tensor:
        """Method to perform inference on test dataloader.

        Args:
            test_loader: The dataloader to be use for testing.
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

        for inp in test_loader:
            op = self._infer_on_batch(inp=inp)
            yield op

    # Perform a sanity check for the forward pass of them model. Ensuring model is defined correctly.
    def perform_sanity_check(self, dl: DataLoader):
        """Method to check if the model forward pass and loss_computation is working or not.

        Args:
            dl: A PyTorch dataloader.
        """
        self._model_to_device()
        x, y = next(iter(dl))
        x, y = self.process_inputs(x=x, y=y)
        op = self._model_forward_pass(x=x)
        loss = self.criterion(op, y)
        print("Sanity Check Completed. Model Forward Pass and Loss Computation Successful")
        print(f"Output Shape : {op.shape}")
        print(f"Loss for a batch :{loss.item()}")

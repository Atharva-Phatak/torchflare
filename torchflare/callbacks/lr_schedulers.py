"""Implements LrScheduler callbacks."""
from abc import ABC
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Union

import torch.optim.lr_scheduler as _schedulers

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class LRSchedulerCallback(Callbacks, ABC):
    """Wrapper class for scheduler callbacks."""

    def __init__(self, scheduler, step_on_batch: bool):
        """Constructor class for Scheduler callback.

        Args:
            scheduler: A pytorch scheduler
            step_on_batch: Whether the scheduler steps after batch or not.
        """
        super(LRSchedulerCallback, self).__init__(order=CallbackOrder.SCHEDULER)
        self._scheduler = scheduler
        self.step_on_batch = step_on_batch
        self.scheduler = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Set scheduler."""
        if self.scheduler is None:
            self.scheduler = self._scheduler(experiment.optimizer)

    def on_batch_end(self, experiment: "Experiment"):
        """Step at end of batch."""
        if self.scheduler is not None and self.step_on_batch:
            self.scheduler.step()

    def on_epoch_end(self, experiment: "Experiment"):
        """Step at the end of epoch."""
        if self.scheduler is not None and not self.step_on_batch:
            if isinstance(self.scheduler, _schedulers.ReduceLROnPlateau):
                val = experiment.exp_logs.get(experiment.val_key + experiment.main_metric)
                self.scheduler.step(val)

            else:
                self.scheduler.step()


class LambdaLR(LRSchedulerCallback, ABC):
    """Multiply learning rate by a factor computed with a given function.
    The function should take int value number of epochs as the only argument.

    Args:
            lr_lambda (function or list of functions): Lambda function for the
                learning rate factor computation.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to False.
    """

    def __init__(
        self,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch: int = -1,
        step_on_batch: bool = False,
    ):
        """Constructor for lambda scheduler."""
        super().__init__(
            lambda opt: _schedulers.LambdaLR(opt, lr_lambda, last_epoch=last_epoch), step_on_batch=step_on_batch
        )


class StepLR(LRSchedulerCallback, ABC):
    """Multiply learning rate by a given factor with a given period.

    Args:
            step_size (int): Period of learning rate update in epochs.
            gamma (float, optional): The multiplicative factor. Defaults to 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to False.
    """

    def __init__(self, step_size: int, gamma: float = 0.1, last_epoch: int = -1, step_on_batch: bool = False):
        """Constructor for StepLR."""
        super().__init__(
            lambda opt: _schedulers.StepLR(opt, step_size, gamma=gamma, last_epoch=last_epoch),
            step_on_batch=step_on_batch,
        )


class MultiStepLR(LRSchedulerCallback, ABC):
    """Multiply learning rate by a given factor on each epoch from a given list.

    Args:
            milestones (list of int): List of epochs number to perform lr step.
            gamma (float, optional): The multiplicative factor. Defaults to 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to False.
    """

    def __init__(
        self, milestones: Iterable[int], gamma: float = 0.1, last_epoch: int = -1, step_on_batch: bool = False
    ):
        """Constructor class for MultiStepLR."""
        super().__init__(
            lambda opt: _schedulers.MultiStepLR(opt, milestones, gamma=gamma, last_epoch=last_epoch),
            step_on_batch=step_on_batch,
        )


class ExponentialLR(LRSchedulerCallback, ABC):
    """Multiply learning rate by a given factor on each epoch.

    Args:
           gamma (float, optional): The multiplicative factor. Defaults to 0.1.
           last_epoch (int): The index of last epoch. Default: -1.
           step_on_batch (bool): Step on each training iteration rather than each epoch.
               Defaults to False.
    """

    def __init__(self, gamma: float, last_epoch: int = -1, step_on_batch: bool = False):
        """Constructor for ExponentialLR."""
        super().__init__(
            lambda opt: _schedulers.ExponentialLR(opt, gamma, last_epoch=last_epoch), step_on_batch=step_on_batch
        )


class CosineAnnealingLR(LRSchedulerCallback, ABC):
    """Set the learning rate of each parameter group using a cosine annealing schedule.

    Args:
            T_max (int): Max number of epochs or iterations.
            eta_min (float, optional): Min learning rate. Defaults to 0.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to True.
    """

    def __init__(self, T_max: int, eta_min: float = 0, last_epoch: int = -1, step_on_batch: bool = True):  # noqa
        """Constructor for CosineAnnealingLR."""
        super().__init__(
            lambda opt: _schedulers.CosineAnnealingLR(opt, T_max, eta_min=eta_min, last_epoch=last_epoch),
            step_on_batch=step_on_batch,
        )


class ReduceLROnPlateau(LRSchedulerCallback, ABC):
    """Reduce learning rate when a metric has stopped improving.


    Args:
            mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored
                has stopped decreasing.In "max" mode it will stop when the quantity monitored has stopped increasing.
            factor (float, optional): The multiplicative factor. Defaults to 0.1.
            patience (int, optional): Number of training epochs without the
                metric improvement to update the learning rate. Defaults to 10.
            verbose (bool, optional): Print info on each update to stdout.
                Defaults to False.
            threshold (float, optional): Threshold for considering the changes
                significant. Defaults to 1e-4.
            threshold_mode (str, optional): Should be 'rel', 'abs'.
                Defaults to 'rel'.
            cooldown (int, optional): Number of epochs to wait before resuming
                normal operation after lr has been updated. Defaults to 0.
            min_lr (float or list of float, optional): Min learning rate.
                Defaults to 0.
            eps (float, optional): Min significant learning rate update.
                Defaults to 1e-8.
    """

    def __init__(
        self,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):
        """Constructor for ReduceLRonPlateau."""
        super().__init__(
            lambda opt: _schedulers.ReduceLROnPlateau(
                opt,
                mode=mode,
                factor=factor,
                patience=patience,
                verbose=verbose,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
            ),
            step_on_batch=False,
        )


class CyclicLR(LRSchedulerCallback, ABC):
    """Sets the learning rate of each parameter group according to cyclical learning rate policy.

    Args:
            base_lr (float or list of float): Initial learning rate.
            max_lr (float or list of float): Max learning rate.
            step_size_up (int, optional): Increase phase duration in epochs or iterations.
                Defaults to 2000.
            step_size_down (int, optional): Decrease phase duration in epochs or iterations.
                Defaults to None.
            mode (str, optional): Should be 'triangular', 'triangular2' or
                'exp_range'. Defaults to 'triangular'.
            gamma (float, optional): Constant for the 'exp_range' policy.
                Defaults to 1.
            scale_fn (function, optional): Custom scaling policy function.
                Defaults to None.
            scale_mode (str, optional): Should be 'cycle' or 'iterations'.
                Defaults to 'cycle'.
            cycle_momentum (bool, optional): Momentum is cycled inversely
                to learning rate between 'base_momentum' and 'max_momentum'.
                Defaults to True.
            base_momentum (float or list of float, optional): Lower momentum
                boundaries in the cycle for each parameter group.
                Defaults to 0.8.
            max_momentum (float or list of float, optional): Upper momentum
                boundaries in the cycle for each parameter group.
                Defaults to 0.9.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to True.
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[float], float]] = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1,
        step_on_batch: bool = True,
    ):
        """Constructor for CyclicLR."""
        super().__init__(
            lambda opt: _schedulers.CyclicLR(
                opt,
                base_lr,
                max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=mode,
                gamma=gamma,
                scale_fn=scale_fn,
                scale_mode=scale_mode,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                last_epoch=last_epoch,
            ),
            step_on_batch=step_on_batch,
        )


class CosineAnnealingWarmRestarts(LRSchedulerCallback, ABC):
    """Set the learning rate of each parameter group using a cosine annealing schedule with a warm restart.

    Args:
           T_0 (int): Number of epochs or iterations for the first restart.
           T_mult (int): T increase factor after a restart.
           eta_min (float, optional): Min learning rate. Defaults to 0.
           last_epoch (int): The index of last epoch. Default: -1.
           step_on_batch (bool): Step on each training iteration rather than each epoch.
               Defaults to True.
    """

    def __init__(
        self, T_0: int, T_mult: int = 1, eta_min: int = 0, last_epoch: int = -1, step_on_batch: bool = True
    ):  # noqa
        """Constructor for CosineAnnealingWarmRestarts."""
        super().__init__(
            lambda opt: _schedulers.CosineAnnealingWarmRestarts(
                opt, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
            ),
            step_on_batch=step_on_batch,
        )


class MultiplicativeLR(LRSchedulerCallback, ABC):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    Args:
            lr_lambda (function or list of functions): A function which computes a
                multiplicative factor given an integer parameter epoch, or a list
                of such functions, one for each group in an optimizer.param_groups.
            last_epoch (int): The index of last epoch. Default: -1.
            step_on_batch (bool): Step on each training iteration rather than each epoch.
                Defaults to False.
    """

    def __init__(
        self,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch: int = -1,
        step_on_batch: bool = False,
    ):
        """Constructor for MultiplicativeLR."""
        super().__init__(
            lambda opt: _schedulers.MultiplicativeLR(opt, lr_lambda, last_epoch=last_epoch),
            step_on_batch=step_on_batch,
        )


class OneCycleLR(LRSchedulerCallback, ABC):
    """Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
    The 1cycle policy anneals
    the learning rate from an initial learning rate to some maximum learning rate
    and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.

    Args:
            max_lr (float or list of float): Upper learning rate boundaries in the
                cycle for each parameter group.
            total_steps (int): The total number of steps in the cycle. Note that
                if a value is not provided here, then it must be inferred by
                providing a value for epochs and steps_per_epoch.
                Defaults to None.
            epochs (int): The number of epochs to train for. This is used along
                with steps_per_epoch in order to infer the total number of steps in
                the cycle if a value for total_steps is not provided.
                Defaults to None.
            steps_per_epoch (int): The number of steps per an epoch to train for. This
                is used along with epochs in order to infer the total number of
                steps in the cycle if a value for total_steps is not provided.
                Defaults to None.
            pct_start (float): The percentage of the cycle (in number of steps)
                spent increasing the learning rate.
                Defaults to 0.3.
            anneal_strategy (str): {'cos', 'linear'}
                Specifies the annealing strategy: "cos" for cosine annealing,
                "linear" for linear annealing.
                Defaults to 'cos'.
            cycle_momentum (bool): If ``True``, momentum is cycled inversely
                to learning rate between 'base_momentum' and 'max_momentum'.
                Defaults to True.
            base_momentum (float or list of float): Lower momentum boundaries in
                the cycle for each parameter group. Note that momentum is cycled
                inversely to learning rate; at the peak of a cycle, momentum is
                'base_momentum' and learning rate is 'max_lr'.
                Defaults to 0.85.
            max_momentum (float or list of float): Upper momentum boundaries in
                the cycle for each parameter group. Functionally,
                it defines the cycle amplitude (max_momentum - base_momentum).
                Note that momentum is cycled inversely
                to learning rate; at the start of a cycle, momentum is
                'max_momentum' and learning rate is 'base_lr'
                Defaults to 0.95.
            div_factor (float): Determines the initial learning rate via
                initial_lr = max_lr/div_factor
                Defaults to 25.
            final_div_factor (float): Determines the minimum learning rate via
                min_lr = initial_lr/final_div_factor
                Defaults to 1e4.
            last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.85,
        max_momentum: Union[float, List[float]] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        """Constructor for OneCycleLR."""
        super().__init__(
            lambda opt: _schedulers.OneCycleLR(
                opt,
                max_lr,
                total_steps=total_steps,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                last_epoch=last_epoch,
            ),
            step_on_batch=True,
        )


__all__ = [
    "LRSchedulerCallback",
    "LambdaLR",
    "OneCycleLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "MultiplicativeLR",
    "MultiStepLR",
    "ReduceLROnPlateau",
    "StepLR",
    "ExponentialLR",
]

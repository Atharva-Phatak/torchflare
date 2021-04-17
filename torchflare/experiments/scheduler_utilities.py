"""Implements scheduler utilities."""
import torch
import transformers

# Structure of dictionary is as follows:
# 'key' -> scheduler , 'step_on_batch' -> whether to step after batch.
# Set True is step is after batch.
scheduler_step = {
    "LambdaLR": False,
    "MultiplicativeLR": False,
    "StepLR": False,
    "MultiStepLR": False,
    "ExponentialLR": False,
    "CosineAnnealingLR": True,
    "ReduceLROnPlateau": False,
    "CyclicLR": True,
    "OneCycleLR": True,
    "CosineAnnealingWarmRestarts": True,
    "get_constant_schedule": True,
    "get_constant_schedule_with_warmup": True,
    "get_cosine_schedule_with_warmup": True,
    "get_cosine_with_hard_restarts_schedule_with_warmup": True,
    "get_linear_schedule_with_warmup": True,
    "get_polynomial_decay_schedule_with_warmup": True,
}


def get_scheduler(scheduler):
    """Method to get scheduler from pytorch/transformers.

    Args:
        scheduler: The scheduler to be used.

    Returns:
        scheduler.

    Raises:
        ValueError: If scheduler is not found raises value error.
    """
    if isinstance(scheduler, str):

        try:
            if scheduler.startswith("get_"):
                sch = getattr(transformers, scheduler.lower())
            else:
                dir_sch = dir(torch.optim.lr_scheduler)
                opts = [o.lower() for o in dir_sch]
                str_idx = opts.index(scheduler.lower())
                sch = getattr(torch.optim.lr_scheduler, dir_sch[str_idx])

            return sch

        except ValueError:
            raise ValueError(
                "Invalid scheduler string input, must match schedulers available in pytorch or transformers"
            )

    elif hasattr(scheduler, "step"):

        return scheduler

    else:

        raise ValueError("Invalid scheduler input")


class LRScheduler:
    """Class around standard scheduler to decide when to step."""

    def __init__(self, scheduler, **kwargs):
        """Constructor method.

        Args:
            scheduler : The scheduler.
            **kwargs: named arguments for a scheduler.
        """
        self.scheduler = get_scheduler(scheduler)(**kwargs)
        self.step_on_batch = scheduler_step[scheduler]
        self.exp = None

    def set_experiment(self, exp):  # noqa

        self.exp = exp

    def _scheduler_step(self):

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val = self.exp.exp_logs.get(self.exp.val_key + self.exp.main_metic)
            self.scheduler.step(val)

        else:
            self.scheduler.step()

    def step(self, current_state):
        """Method to perform the scheduler step.

        Args:
            current_state: The current state of experiment.
        """
        if self.step_on_batch and "batch" in current_state.value:
            self._scheduler_step()

        elif self.step_on_batch is False and "epoch" in current_state.value:
            self._scheduler_step()

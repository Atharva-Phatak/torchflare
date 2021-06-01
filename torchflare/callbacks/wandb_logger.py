"""Implements logger for weights and biases."""
from abc import ABC
from typing import TYPE_CHECKING, Dict, List, Optional

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder
from torchflare.utils.imports_check import module_available

_AVAILABLE = module_available("wandb")
if _AVAILABLE:
    import wandb
else:
    wandb = None


if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class WandbLogger(Callbacks, ABC):
    """Callback to log your metrics and loss values to  wandb platform.

    For more information about wandb take a look at [Weights and Biases](https://wandb.ai/)

    Args:
            project: The name of the project where you're sending the new run
            entity:  An entity is a username or team name where you're sending runs.
            name: A short display name for this run
            config: The hyperparameters for your model and experiment as a dictionary
            tags:  List of strings.
            directory: where to save wandb local run directory.
                If set to None it will use experiments save_dir argument.
            notes: A longer description of the run, like a -m commit message in git

    Note:
            set os.environ['WANDB_SILENT'] = True to silence wandb log statements.
            If this is set all logs will be written to WANDB_DIR/debug.log

    Examples:
        .. code-block::

            from torchflare.callbacks import WandbLogger

            params = {"bs": 16, "lr": 0.3}

            logger = WandbLogger(
                project="Experiment",
                entity="username",
                name="Experiment_10",
                config=params,
                tags=["Experiment", "fold_0"])
    """

    def __init__(
        self,
        project: str,
        entity: str,
        name: str = None,
        config: Dict = None,
        tags: List[str] = None,
        notes: Optional[str] = None,
        directory: str = None,
    ):
        """Constructor of WandbLogger."""
        super(WandbLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.entity = entity
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.notes = notes
        self.dir = directory
        self.experiment = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Experiment start."""
        self.experiment = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            dir=self.dir,
        )

    def on_epoch_end(self, experiment: "Experiment"):
        """Method to log metrics and values at the end of very epoch."""
        logs = {k: v for k, v in experiment.exp_logs.items() if k != experiment.epoch_key}
        self.experiment.log(logs)

    def on_experiment_end(self, experiment: "Experiment"):
        """Method to end experiment after training is done."""
        self.experiment.finish()

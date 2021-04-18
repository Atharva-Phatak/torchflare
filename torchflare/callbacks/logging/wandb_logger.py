"""Implements logger for weights and biases."""
from abc import ABC
from typing import Dict, List, Optional

import wandb

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class WandbLogger(Callbacks, ABC):
    """Callback to log your metrics and loss values to  wandb platform.

    For more information about wandb take a look at [Weights and Biases](https://wandb.ai/)
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
        """Constructor of WandbLogger.

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

        """
        super(WandbLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.experiment = wandb.init(
            entity=entity, project=project, name=name, config=config, tags=tags, notes=notes, dir=directory
        )

    def epoch_end(self, epoch, logs):
        """Method to log metrics and values at the end of very epoch.

        Args:
            logs: A dictionary containing metrics and loss values.
            epoch: The current epoch
        """
        _ = logs.pop("Time")
        self.experiment.log(logs)

    def experiment_end(self):
        """Method to end experiment after training is done."""
        self.experiment.finish()

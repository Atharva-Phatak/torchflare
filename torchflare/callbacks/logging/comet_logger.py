"""Implements Comet Logger."""

from abc import ABC
from typing import List

import comet_ml

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder


class CometLogger(Callbacks, ABC):
    """Callback to log your metrics and loss values to Comet to track your experiments.

    For more information about Comet look at [Comet.ml](https://www.comet.ml/site/)
    """

    def __init__(
        self, api_token: str, params: dict, project_name: str, workspace: str, tags: List[str],
    ):
        """Constructor for CometLogger class.

        Args:
            api_token: Your API key obtained from comet.ml
            params: The hyperparameters for your model and experiment as a dictionary
            project_name: Send your experiment to a specific project.
                    Otherwise, will be sent to Uncategorized Experiments.
            workspace: Attach an experiment to a project that belongs to this workspace
            tags: List of strings.
        """
        super(CometLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.api_token = api_token
        self.project_name = project_name
        self.workspace = workspace
        self.params = params
        self.tags = tags

        self.experiment = comet_ml.Experiment(
            project_name=self.project_name,
            api_key=self.api_token,
            workspace=self.workspace,
            log_code=False,
            display_summary_level=0,
        )

        if self.tags is not None:
            self.experiment.add_tags(self.tags)

        if self.params is not None:
            self.experiment.log_parameters(self.params)

    def epoch_end(self, epoch: int, logs: dict):
        """Function to log your metrics and values at the end of very epoch.

        Args:
            logs : A dictionary containing metrics and loss values.
            epoch: The current epoch
        """
        _ = logs.pop("Time")
        self.experiment.log_metrics(logs, step=epoch)

    def experiment_end(self):
        """Function to close the experiment when training ends."""
        self.experiment.end()

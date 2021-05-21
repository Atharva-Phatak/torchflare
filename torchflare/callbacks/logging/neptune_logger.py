"""Implements Neptune Logger."""
from abc import ABC
from typing import TYPE_CHECKING, List

import neptune.new as neptune

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class NeptuneLogger(Callbacks, ABC):
    """Callback to log your metrics and loss values to Neptune to track your experiments.

    For more information about Neptune take a look at  [Neptune](https://neptune.ai/)
    """

    def __init__(
        self,
        project_dir: str,
        api_token: str,
        params: dict = None,
        experiment_name: str = None,
        tags: List[str] = None,
    ):
        """Constructor for NeptuneLogger Class.

        Args:
            project_dir: The qualified name of a project in a form of namespace/project_name
            params: he hyperparameters for your model and experiment as a dictionary
            experiment_name: The name of the experiment
            api_token: User’s API token
            tags:  List of strings.
        """
        super(NeptuneLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.project_dir = project_dir
        self.api_token = api_token
        self.params = params
        self.tags = tags
        self.experiment_name = experiment_name
        self.experiment = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
        self.experiment = neptune.init(
            project=self.project_dir, api_token=self.api_token, tags=self.tags, name=self.experiment_name
        )
        self.experiment["params"] = self.params

    def _log_metrics(self, name, value, epoch):

        self.experiment[name].log(value=value, step=epoch)

    def on_epoch_end(self, experiment: "Experiment"):
        """Method to log metrics and values at the end of very epoch."""
        for key, value in experiment.exp_logs.items():
            if key != experiment.epoch_key:
                epoch = experiment.exp_logs[experiment.epoch_key]
                self._log_metrics(name=key, value=value, epoch=epoch)

    def on_experiment_end(self, experiment: "Experiment"):
        """Method to end experiment after training is done."""
        self.experiment.stop()
        self.experiment = None

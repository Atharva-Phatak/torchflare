"""Implements Comet Logger."""

from abc import ABC
from typing import TYPE_CHECKING, List

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder
from torchflare.utils.imports_check import module_available

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment

_AVAILABLE = module_available("come_ml")
if _AVAILABLE:
    import comet_ml
else:
    comet_ml = None


class CometLogger(Callbacks, ABC):
    """Callback to log your metrics and loss values to Comet to track your experiments.
    For more information about Comet look at [Comet.ml](https://www.comet.ml/site/)

    Args:
        api_token: Your API key obtained from comet.ml
        params: The hyperparameters for your model and experiment as a dictionary
        project_name: Send your experiment to a specific project.
                Otherwise, will be sent to Uncategorized Experiments.
        workspace: Attach an experiment to a project that belongs to this workspace
        tags: List of strings.

    Examples:
        .. code-block::

            from torchflare.callbacks import CometLogger

            params = {"bs": 16, "lr": 0.3}

            logger = CometLogger(
                project_name="experiment_10",
                workspace="username",
                params=params,
                tags=["Experiment", "fold_0"],
                api_token="your_secret_api_token",
            )
    """

    def __init__(
        self,
        api_token: str,
        params: dict,
        project_name: str,
        workspace: str,
        tags: List[str],
    ):
        """Constructor for CometLogger class."""
        super(CometLogger, self).__init__(order=CallbackOrder.LOGGING)
        self.api_token = api_token
        self.project_name = project_name
        self.workspace = workspace
        self.params = params
        self.tags = tags
        self.experiment = None

    def on_experiment_start(self, experiment: "Experiment"):
        """Start of experiment."""
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

    def on_epoch_end(self, experiment: "Experiment"):
        """Function to log your metrics and values at the end of very epoch."""
        logs = {k: v for k, v in experiment.exp_logs.items() if k != experiment.epoch_key}
        self.experiment.log_metrics(logs, step=experiment.exp_logs[experiment.epoch_key])

    def on_experiment_end(self, experiment: "Experiment"):
        """Function to close the experiment when training ends."""
        self.experiment.end()
        self.experiment = None

# flake8: noqa


import os

from torchflare.callbacks.wandb_logger import WandbLogger
from torchflare.callbacks.tensorboard_logger import TensorboardLogger
from torchflare.callbacks.neptune_logger import NeptuneLogger
from unittest.mock import patch

os.environ["WANDB_SILENT"] = "true"
# os.environ["NEPTUNE_API_TOKEN"] = "Dummy_token"
os.environ["WB_API_TOKEN"] = "Dummy_wandb_token"


class TestExperiment:
    def __init__(self):
        pass

    @patch("torchflare.callbacks.logging.neptune_logger.neptune")
    def test_neptune_mock(self, neptune):
        """Simple to check if same experiment is created."""
        logger = NeptuneLogger(api_token="test", project_dir="namespace/project")
        logger.on_experiment_start(self)
        created_experiment = neptune.init(name="namespace/project", api_token="test")
        logger.on_epoch_end(self)
        logger.on_experiment_end(self)
        assert logger.experiment is not None
        assert created_experiment.name == logger.experiment.name
        assert created_experiment.id == logger.experiment.id
        assert created_experiment.on_experiment_start.assert_called_once()
        assert created_experiment.on_epoch_end.assert_called_once()
        assert created_experiment.on_experiment_end.assert_called_once()

    @patch("torchflare.callbacks.logging.wandb_logger.wandb")
    def test_wandb_mock(self, wandb):
        """Simple test to check if same experiment is created or not."""
        logger = WandbLogger(project="test", entity="project")
        logger.on_experiment_start(self)
        wandb.init.assert_called_once(self)
        wandb_exp = wandb.init(project="test", entity="project")
        logger.on_epoch_end(self)
        logger.on_experiment_end(self)
        assert logger.experiment is not None
        assert logger.experiment.entity == wandb_exp.entity
        assert logger.experiment.id == wandb_exp.id
        assert wandb_exp.on_experiment_start.assert_called_once()
        assert wandb_exp.on_epoch_end.assert_called_once()
        assert wandb_exp.on_experiment_end.assert_called_once()


class Experiment:
    def __init__(self, logger):
        self.epoch_key = "epoch"
        self.exp_logs = None
        self.logger = logger


    def start_log(self):
        acc = 10
        f1 = 10
        loss = 100
        self.logger.on_experiment_start(self)
        for epoch in range(10):
            self.exp_logs = {"epoch": epoch, "acc": acc, "f1": f1, "loss": loss, "Time": 5}
            acc += 10
            f1 += 10

            loss = loss / 10
            self.logger.on_epoch_end(self)

        self.logger.on_experiment_end(self)

"""
def test_neptune():
    params = {"bs": 16, "lr": 0.01}
    logger = NeptuneLogger(
            project_dir="torchflare456/torchflare-tests",
            params=params,
            experiment_name="Neptune_test",
            tags=["Dummy", "test"],
            api_token="ANONYMOUS",)
    neptune_exp = Experiment(logger)
    neptune_exp.start_log()

"""
"""
@pytest.mark.skip(reason="Callback running correctly. Will need seprate API token for general Tests.")
def test_wandb(tmpdir):
    params = {"bs": 16, "lr": 0.01}
    logger = WandbLogger(
        project="dl-experiment",
        entity="notsogenius",
        name="dummy_exp_2",
        config=params,
        tags=["Dummy", "test"],
        directory=tmpdir.mkdir("/callbacks/"),
    )

    acc = 10
    f1 = 10
    loss = 100
    for epoch in range(10):
        d = {"acc": acc, "f1": f1, "loss": loss, "Time": 5}
        acc += 10
        f1 += 10

        loss = loss / 10
        logger.on_epoch_end(epoch=epoch, logs=d)

    logger.on_experiment_end()
"""




def test_tensorboard(tmpdir):
    l = TensorboardLogger(log_dir=tmpdir.mkdir("/callbacks"))
    ob = Experiment(logger=l)
    ob.start_log()

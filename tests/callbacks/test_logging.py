# flake8: noqa


import os

from torchflare.callbacks.logging.wandb_logger import  WandbLogger
from torchflare.callbacks.logging.tensorboard_logger import  TensorboardLogger
from torchflare.callbacks.logging.neptune_logger import NeptuneLogger
from unittest.mock import patch

os.environ["WANDB_SILENT"] = "true"
# os.environ["NEPTUNE_API_TOKEN"] = "Dummy_token"
os.environ["WB_API_TOKEN"] = "Dummy_wandb_token"


@patch("torchflare.callbacks.logging.neptune_logger.neptune")
def test_neptune_mock(neptune):
    """Simple to check if same experiment is created."""
    logger = NeptuneLogger(api_token="test", project_dir="namespace/project")
    created_experiment = neptune.init(name="namespace/project", api_token="test")
    assert logger.experiment is not None
    assert created_experiment.name == logger.experiment.name
    assert created_experiment.id == logger.experiment.id


@patch("torchflare.callbacks.logging.wandb_logger.wandb")
def test_wandb_mock(wandb):
    """Simple test to check if same experiment is created or not."""
    logger = WandbLogger(project="test", entity="project")
    wandb.init.assert_called_once()
    wandb_exp = wandb.init(project="test", entity="project")
    assert logger.experiment is not None
    assert logger.experiment.entity == wandb_exp.entity
    assert logger.experiment.id == wandb_exp.id


""""
@pytest.mark.skip(reason="Callback running correctly.Requires secret api token.")
def test_neptune():

    params = {"bs": 16, "lr": 0.01}
    logger = NeptuneLogger(
        project_dir="notsogenius/DL-Experiments",
        params=params,
        experiment_name="Neptune_test",
        tags=["Dummy", "test"],
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    )
    cb = CallbackRunner(callbacks=[logger])
    acc = 10
    f1 = 10
    loss = 10

    cb(current_state=ExperimentStates.EXP_START)
    for epoch in range(10):
        d = {"acc": acc, "f1": f1, "loss": loss, "Time": 5}
        acc += 10
        f1 += 10

        loss = loss / 10

        cb(current_state=ExperimentStates.EPOCH_END, epoch=epoch, logs=d)
    cb(current_state=ExperimentStates.EXP_END)


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
        logger.epoch_end(epoch=epoch, logs=d)

    logger.experiment_end()
"""


def test_tensorboard(tmpdir):
    logger = TensorboardLogger(log_dir=tmpdir.mkdir("/callbacks"))
    acc = 10
    f1 = 10
    loss = 100
    for epoch in range(10):
        d = {"acc": acc, "f1": f1, "loss": loss, "Time": 5}
        acc += 10
        f1 += 10

        loss = loss / 10
        logger.epoch_end(epoch=epoch, logs=d)

    logger.experiment_end()

# flake8: noqa

import os

import torch

from torchflare.callbacks.load_checkpoint import LoadCheckpoint
from torchflare.callbacks.model_checkpoint import ModelCheckpoint
from torchflare.core.state import State


class Experiment:
    def __init__(self, model, optimizer, cbs):

        self.state = State(model=model, optimizer=optimizer)  # Dummy model just to see if the checkpoint callback works
        # self.save_dir = save_dir
        # self.model_name = "DummyModel.bin"
        self.scheduler = None
        self._stop_training = False
        self._model_state = None
        self.cb = cbs

        self.device = "cpu"
        self.exp_logs = {}
        self.epoch_key = "Epoch"
        # if not os.path.exists(self.save_dir):
        # os.mkdir(self.save_dir)

        self.path = cbs.path
        # assert os.path.exists(self.save_dir)
        self.scheduler_stepper = None
        # print(type(self.path))
        self.cb_lc = LoadCheckpoint(self.path)
        # callback runner for load_checkpoint. Since we test model only after training done.

    def fit(self):

        train_loss = 0.01
        val_loss = 0.02
        train_acc = 10
        val_acc = 10


        for epoch in range(10):

            train_loss += 0.01
            val_loss -= 0.02
            train_acc += 0.2
            val_acc += 0.2

            # print(epoch)
            logs = {
                "Epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
            }

            self.exp_logs.update(logs)
            self.cb.on_epoch_end(self)

            if self._stop_training:
                break
        self.cb.on_experiment_end(self)
        assert os.path.exists(self.path)

    def check_checkpoints(self):

        assert os.path.exists(self.path)
        ckpt = torch.load(self.path)
        model_dict = self.state.model.state_dict()
        self.cb_lc.on_experiment_start(self)
        for layer_name, weight in ckpt["model_state_dict"].items():
            assert layer_name in model_dict
            assert torch.all(model_dict[layer_name] == weight)

        assert self.state.optimizer.state_dict() == ckpt["optimizer_state_dict"]

    def check_checkpoints_dict(self):

        assert os.path.exists(self.path)
        ckpt = torch.load(self.path)

        self.cb_lc.on_experiment_start(self)
        for model_k in ckpt["model_state_dict"]:
            model_dict = self.state.model[model_k].state_dict()
            for layer_name, weight in self.state.model[model_k].state_dict().items():
                assert layer_name in model_dict
                assert torch.all(model_dict[layer_name] == weight)
        for optimizer_key, optimizer in self.state.optimizer.items():
            assert optimizer.state_dict() == ckpt["optimizer_state_dict"][optimizer_key]


def test_checkpoint_on_loss(tmpdir):

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    mckpt = ModelCheckpoint(
        mode="min", monitor="val_loss", save_dir=tmpdir.mkdir("/callbacks"), file_name="DummyModel.bin"
    )
    trainer = Experiment(model=model, cbs=mckpt, optimizer=optimizer)

    trainer.fit()
    trainer.check_checkpoints()
    assert os.path.exists(trainer.path) is True


def test_checkpoint_on_acc(tmpdir):

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    mckpt = ModelCheckpoint(
        mode="max", monitor="val_acc", save_dir=tmpdir.mkdir("/callbacks"), file_name="DummyModel.bin"
    )
    trainer = Experiment(model=model, cbs=mckpt, optimizer=optimizer)

    trainer.fit()
    trainer.check_checkpoints()
    assert os.path.exists(trainer.path) is True


def test_checkpoint_multiple_models(tmpdir):

    model = {"model_A": torch.nn.Linear(10, 2), "model_B": torch.nn.Linear(10, 2)}
    optimizer = {
        "optimizer_A": torch.optim.SGD(model["model_A"].parameters(), lr=3e-4),
        "optimizer_B": torch.optim.SGD(model["model_B"].parameters(), lr=3e-4),
    }
    mckpt = ModelCheckpoint(
        mode="min", monitor="val_loss", save_dir=tmpdir.mkdir("/callbacks"), file_name="DummyModel.bin"
    )
    trainer = Experiment(model=model, cbs=mckpt, optimizer=optimizer)

    trainer.fit()
    trainer.check_checkpoints_dict()
    assert os.path.exists(trainer.path) is True


# test_checkpoint_on_acc()

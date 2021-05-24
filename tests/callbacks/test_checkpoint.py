# flake8: noqa

import os

import torch

from torchflare.callbacks.load_checkpoint import LoadCheckpoint
from torchflare.callbacks.model_checkpoint import ModelCheckpoint


class Experiment:
    def __init__(self, model, cbs):

        self.model = model  # Dummy model just to see if the checkpoint callback works
        # self.save_dir = save_dir
        # self.model_name = "DummyModel.bin"
        self.scheduler = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-4)
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

        self.cb.on_experiment_start(self)

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
        model_dict = self.model.state_dict()
        self.cb_lc.on_experiment_start(self)
        for layer_name, weight in ckpt["model_state_dict"].items():
            assert layer_name in model_dict
            assert torch.all(model_dict[layer_name] == weight)

        assert self.optimizer.state_dict() == ckpt["optimizer_state_dict"]


def test_checkpoint_on_loss(tmpdir):

    model = torch.nn.Linear(10, 2)
    mckpt = ModelCheckpoint(
        mode="min", monitor="val_loss", save_dir=tmpdir.mkdir("/callbacks"), file_name="DummyModel.bin"
    )
    trainer = Experiment(model=model, cbs=mckpt)

    trainer.fit()
    trainer.check_checkpoints()
    assert os.path.exists(trainer.path) is True


def test_checkpoint_on_acc(tmpdir):

    model = torch.nn.Linear(10, 2)
    mckpt = ModelCheckpoint(
        mode="max", monitor="val_acc", save_dir=tmpdir.mkdir("/callbacks"), file_name="DummyModel.bin"
    )
    trainer = Experiment(model=model, cbs=mckpt)

    trainer.fit()
    trainer.check_checkpoints()
    assert os.path.exists(trainer.path) is True


# test_checkpoint_on_acc()

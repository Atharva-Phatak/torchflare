# flake8: noqa
from torchflare.callbacks.early_stopping import EarlyStopping


class Experiment:
    def __init__(self, cbs):

        self.stop_training = False
        self._model_state = None
        self.cb = cbs
        self.exp_logs = {}


    def fit(self):

        train_loss = 0.01
        val_loss = 0.02
        train_acc = 10
        val_acc = 10

        self.cb.on_experiment_start(self)

        for epoch in range(10):

            train_loss += 0.01
            val_loss += 0.02
            train_acc += 0.2
            val_acc -= 0.2

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

            if self.stop_training:
                break
        self.cb.on_experiment_end(self)


def test_on_val_loss():

    es = EarlyStopping(mode="min",monitor = "val_loss")
    trainer = Experiment(cbs=es)

    trainer.fit()

    assert es.monitor == "val_loss"
    assert trainer.stop_training is True


def test_on_metric():
    es = EarlyStopping(monitor="val_acc", mode="max")
    trainer = Experiment(es)

    trainer.fit()
    assert es.monitor == "val_acc"
    assert trainer.stop_training is True

# flake8: noqa
import time
from torchflare.callbacks.model_history import History


class Experiment:
    def __init__(self):

        self._stop_training = False
        self._model_state = None
        self.save_dir = None
        self.cb = History()
        self.history = None
        self.exp_logs = {}

    def fit(self):

        train_loss = 0.01
        val_loss = 0.02
        train_acc = 10
        val_acc = 10

        self.cb.on_experiment_start(self)

        for epoch in range(10):

            start = time.time()
            train_loss += 0.01
            val_loss += 0.02
            train_acc += 0.2
            val_acc += 0.2

            time.sleep(0.1)
            # print(epoch)
            logs = {
                "Epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "Time": (time.time() - start),
            }

            self.exp_logs.update(logs)
            self.cb.on_epoch_end(self)

            if self._stop_training:
                break
        self.cb.on_experiment_end(self)


def test_history(tmpdir):

    hist = History()
    trainer = Experiment()
    trainer.save_dir = tmpdir.mkdir("/callbacks")
    trainer.fit()
    # print(trainer.history.history)
    assert isinstance(trainer.history, dict) is True

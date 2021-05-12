# flake8: noqa
import time
from torchflare.callbacks.callback import CallbackRunner
from torchflare.callbacks.states import ExperimentStates
from torchflare.callbacks.model_history import History


class DummyPipeline:
    def __init__(self, cbs):

        self._stop_training = False
        self._model_state = None
        self.save_dir = None
        self.history = cbs[0]
        self.cb = CallbackRunner(cbs)

        self.cb.set_experiment(self)
        self.exp_logs = {}

    @property
    def set_model_state(self):

        return self._model_state

    @set_model_state.setter
    def set_model_state(self, state):

        self._model_state = state
        if self.cb is not None:
            self.cb(current_state=self._model_state)

    def fit(self):

        train_loss = 0.01
        val_loss = 0.02
        train_acc = 10
        val_acc = 10

        self.set_model_state = "on_experiment_start"

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
            self.set_model_state = "on_epoch_end"

            if self._stop_training:
                break
        self.set_model_state = "on_experiment_end"


def test_history(tmpdir):

    hist = History()
    trainer = DummyPipeline(cbs=[hist])
    trainer.save_dir = tmpdir.mkdir("/callbacks")
    trainer.fit()
    # print(trainer.history.history)
    assert isinstance(trainer.history, dict) is True

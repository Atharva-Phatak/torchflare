# flake8: noqa
import pytest
import torch
import torchflare.callbacks as cbs
from torchmetrics import Accuracy
from torchflare.experiments.experiment import Experiment
from torchflare.utils.seeder import seed_all
from torchflare.experiments.config import ModelConfig
from torchflare.core.state import State
import os
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.model = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def create_ds(args):
    dataset = torch.utils.data.TensorDataset(*args) if len(args) > 1 else args[0]
    return dataset


def test_experiment(tmpdir):
    optimizer = "Adam"
    optimizer_lr = 3e-4
    criterion = "cross_entropy"
    fp16 = True if torch.cuda.is_available() else False
    num_samples, num_features, num_classes = int(1e4), int(100), 4
    test_samples = 50

    config = ModelConfig(
        nn_module=Model,
        module_params=dict(num_features=num_features, num_classes=num_classes),
        optimizer=optimizer,
        criterion=criterion,
        optimizer_params=dict(lr=optimizer_lr),
    )

    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, num_classes, size=(num_samples,))
    test_data = torch.randn(test_samples, num_features)

    def _test_fit_loader(device):

        seed_all(42)
        save_dir = tmpdir.mkdir("/test_save")
        file_name = "test_classification.bin"
        dataset = torch.utils.data.TensorDataset(X, y)
        test_dataset = create_ds((test_data,))
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16)

        metric_list = [
            Accuracy(num_classes=num_classes),
        ]

        callbacks = [
            cbs.EarlyStopping(monitor="val_accuracy", mode="max"),
            cbs.ModelCheckpoint(monitor="val_accuracy", mode="max", save_dir=save_dir, file_name=file_name),
            cbs.CosineAnnealingWarmRestarts(T_0=2),
        ]

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )

        exp.compile_experiment(
            model_config=config,
            metrics=metric_list,
            callbacks=callbacks,
            main_metric="accuracy",
        )

        assert isinstance(exp.state, State) is True
        exp.fit_loader(train_dl=loader, valid_dl=loader)
        logs = exp.get_logs()
        assert isinstance(logs, pd.DataFrame) is True
        outputs = []
        for op in exp.predict_on_loader(
            test_dl=test_dl, device=device, path_to_model=os.path.join(save_dir, file_name)
        ):
            outputs.extend(op)

        assert len(outputs) == test_samples

    def _test_fit(device):
        seed_all(42)
        save_dir = tmpdir.mkdir("/test_saves")
        file_name = "test_classification.bin"

        metric_list = [
            Accuracy(num_classes=num_classes),
        ]

        callbacks = [
            cbs.EarlyStopping(monitor="val_accuracy", mode="max"),
            cbs.ModelCheckpoint(monitor="val_accuracy", mode="max", save_dir=save_dir, file_name=file_name),
            cbs.CosineAnnealingWarmRestarts(T_0=2),
        ]

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )

        exp.compile_experiment(
            model_config=config,
            metrics=metric_list,
            callbacks=callbacks,
            main_metric="accuracy",
        )
        assert isinstance(exp.state, State) is True
        exp.fit(x=X, y=y, val_data=(X, y), batch_size=32)
        logs = exp.get_logs()
        assert isinstance(logs, pd.DataFrame) is True
        outputs = []
        for op in exp.predict(
            x=test_data, device=device, path_to_model=os.path.join(save_dir, file_name), batch_size=16
        ):
            outputs.extend(op)

        assert len(outputs) == test_samples

    def _test_dict_inputs(device):

        seed_all(42)
        exp_config = ModelConfig(
            nn_module={"model_A": Model, "model_B": Model},
            module_params={
                "model_A": {"num_features": num_features, "num_classes": num_classes},
                "model_B": {"num_features": num_features, "num_classes": num_classes},
            },
            optimizer={"model_A": optimizer, "model_B": optimizer},
            optimizer_params={"model_A": {"lr": optimizer_lr}, "model_B": {"lr": optimizer_lr}},
            criterion=criterion,
        )

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )
        exp.compile_experiment(
            model_config=exp_config,
            metrics=None,
            callbacks=None,
            main_metric="val_loss",
        )
        assert isinstance(exp.state, State) is True
        assert len(exp.state.model.keys()) == 2
        assert len(exp.state.optimizer.keys()) == 2

    def _test_dict_inputs_error(device):

        seed_all(42)
        exp_config = ModelConfig(
            nn_module={"model_A": Model, "model_B": Model},
            module_params={
                "model_A": {"num_features": num_features, "num_classes": num_classes},
                "model_B": {"num_features": num_features, "num_classes": num_classes},
            },
            optimizer=optimizer,
            optimizer_params={"lr": optimizer_lr},
            criterion=criterion,
        )

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )
        with pytest.raises(ValueError):
            exp.compile_experiment(
                model_config=exp_config,
                metrics=None,
                callbacks=None,
                main_metric="val_loss",
            )

    def _test_config_error():
        with pytest.raises(ValueError):
            exp_config = ModelConfig(
                nn_module={"model_A": Model, "model_B": Model},
                module_params={
                    "model_A": {"num_features": num_features, "num_classes": num_classes},
                    "model_B": {"num_features": num_features, "num_classes": num_classes},
                },
                optimizer={"optim_A": optimizer, "optim_B": optimizer},
                optimizer_params={"optim_A": {"lr": optimizer_lr}, "optim_B": {"lr": optimizer_lr}},
                criterion=criterion,
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _test_fit(device=device)
    _test_fit_loader(device=device)
    _test_dict_inputs(device=device)
    _test_config_error()
    _test_dict_inputs_error(device=device)

# flake8: noqa
import torch
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
from torchflare.experiments.experiment import Experiment
from torchflare.utils.seeder import seed_all
import os


def test_experiment(tmpdir):
    optimizer = "Adam"
    optimizer_params = dict(lr=1e-4)
    criterion = "cross_entropy"
    fp16 = True if torch.cuda.is_available() else False
    num_samples, num_features, num_classes = int(1e4), int(100), 4
    test_samples = 50
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, num_classes, size=(num_samples,))
    test_data = torch.randn(test_samples, num_features)

    def _test_fit_loader(device):

        seed_all(42)
        save_dir = tmpdir.mkdir("/test_save")
        file_name = "test_classification.bin"
        dataset = torch.utils.data.TensorDataset(X, y)
        test_dataset = torch.utils.data.TensorDataset(test_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=16)
        model = torch.nn.Linear(num_features, num_classes)

        metric_list = [
            metrics.Accuracy(num_classes=num_classes, multilabel=False),
            metrics.F1Score(num_classes=num_classes, multilabel=False),
        ]

        callbacks = [
            cbs.EarlyStopping(monitor="accuracy", mode="max"),
            cbs.ModelCheckpoint(monitor="accuracy", mode="max", save_dir=save_dir, file_name=file_name),
            cbs.CosineAnnealingWarmRestarts(T_0=2),
        ]

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )

        exp.compile_experiment(
            model=model,
            metrics=metric_list,
            callbacks=callbacks,
            main_metric="accuracy",
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            criterion=criterion,
        )
        exp.fit_loader(train_dl=loader, valid_dl=loader)
        exp.plot_history(keys=["accuracy"], save_fig=False, plot_fig=False)
        outputs = []
        for op in exp.predict_on_loader(test_dl=test_dl, path_to_model=os.path.join(save_dir, file_name)):
            outputs.extend(op)

        assert len(outputs) == test_samples

    def _test_fit(device):
        seed_all(42)
        save_dir = tmpdir.mkdir("/test_saves")
        file_name = "test_classification.bin"
        model = torch.nn.Linear(num_features, num_classes)

        metric_list = [
            metrics.Accuracy(num_classes=num_classes, multilabel=False),
            metrics.F1Score(num_classes=num_classes, multilabel=False),
        ]

        callbacks = [
            cbs.EarlyStopping(monitor="accuracy", mode="max"),
            cbs.ModelCheckpoint(monitor="accuracy", mode="max", save_dir=save_dir, file_name=file_name),
            cbs.CosineAnnealingWarmRestarts(T_0=2),
        ]

        exp = Experiment(
            num_epochs=10,
            fp16=fp16,
            device=device,
            seed=42,
        )

        exp.compile_experiment(
            model=model,
            metrics=metric_list,
            callbacks=callbacks,
            main_metric="accuracy",
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            criterion=criterion,
        )
        exp.fit(x=X, y=y, val_data=(X, y), batch_size=32)
        exp.plot_history(keys=["accuracy"], save_fig=False, plot_fig=False)
        outputs = []
        for op in exp.predict(x=test_data, path_to_model=os.path.join(save_dir, file_name), batch_size=16):
            outputs.extend(op)

        assert len(outputs) == test_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _test_fit(device=device)
    _test_fit_loader(device=device)

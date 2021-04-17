# flake8: noqa
import torch
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
from torchflare.experiments.experiment import Experiment
from torchflare.utils.seeder import seed_all


def test_experiment(tmpdir):
    optimizer = "Adam"
    optimizer_params = dict(lr=1e-4)
    scheduler = "ReduceLROnPlateau"
    scheduler_params = dict(mode="max")
    criterion = "cross_entropy"
    fp16 = True if torch.cuda.is_available() else False

    def _test(device):

        seed_all(42)
        num_samples, num_features, num_classes = int(1e4), int(100), 4
        X = torch.rand(num_samples, num_features)
        y = torch.randint(0, num_classes, size=(num_samples,))

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = torch.nn.Linear(num_features, num_classes)

        metric_list = [
            metrics.Accuracy(num_classes=num_classes, multilabel=False),
            metrics.F1Score(num_classes=num_classes, multilabel=False),
        ]

        callbacks = [cbs.EarlyStopping(monitor="accuracy", mode="max"), cbs.ModelCheckpoint(monitor="accuracy")]

        exp = Experiment(
            num_epochs=10,
            save_dir=tmpdir.mkdir("/test_save"),
            model_name="test_classification.bin",
            fp16=fp16,
            device=device,
            seed=42,
            using_batch_mixers=False,
            compute_train_metrics=False,
        )

        exp.compile_experiment(
            model=model,
            metrics=metric_list,
            callbacks=callbacks,
            main_metric="accuracy",
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            criterion=criterion,
        )
        exp.run_experiment(train_dl=loader, valid_dl=loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _test(device=device)

import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

import torchflare.callbacks as cbs
from torchflare.experiments import Experiment, ModelConfig

# Defining PyTorch Model.


class Net(nn.Module):
    def __init__(self, n_classes, p_dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=p_dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    # Defining Dataloaders.

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_mnist_dataset = MNIST(
        download=True, root="mnist_data", transform=data_transform, train=True
    )
    val_mnist_dataset = MNIST(
        download=False, root="mnist_data", transform=data_transform, train=False
    )
    train_dl = DataLoader(train_mnist_dataset, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_mnist_dataset, batch_size=128, shuffle=False)

    # Defining callbacks.
    callbacks = [
        cbs.EarlyStopping(monitor="accuracy", mode="max"),
        cbs.ModelCheckpoint(monitor="accuracy", mode="max"),
        cbs.ReduceLROnPlateau(mode="max", patience=3),
    ]
    # Defining metrics.
    metric_list = [torchmetrics.Accuracy(num_classes=10)]

    # Define some params for the experiment
    exp = Experiment(num_epochs=10, fp16=True, device="cuda", seed=42)

    # Define model config
    config = ModelConfig(
        nn_module=Net,
        module_params={"n_classes": 10, "p_dropout": 0.3},
        optimizer="Adam",
        optimizer_params={"lr": 3e-4},
        criterion="cross_entropy",
    )

    # Compile the experiment
    exp.compile_experiment(
        model_config=config, callbacks=callbacks, metrics=metric_list, main_metric="accuracy"
    )

    # Run the experiment
    exp.fit_loader(train_dl=train_dl, valid_dl=val_dl)

    # Get logs for the experiment
    logs = exp.get_logs()

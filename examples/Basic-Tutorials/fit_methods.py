import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

import torchflare.callbacks as cbs
from torchflare.experiments import Experiment, ModelConfig

# Defining the Model


class Net(torch.nn.Module):
    def __init__(self, out_features):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(7 * 7 * 64, 256),
            nn.BatchNorm1d(num_features=256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, out_features),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the 3 last dimensions (channels, width, height) to one
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Reading the dataset

    train_df = pd.read_csv("dataset/train.csv")

    classes = train_df.label.nunique()

    train_labels = train_df["label"].values
    train_images = train_df.iloc[:, 1:].values.astype("float32")
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, stratify=train_labels, random_state=123, test_size=0.20
    )

    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    val_images = val_images.reshape(val_images.shape[0], 1, 28, 28)

    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Defining Metrics
    metric_list = [Accuracy(num_classes=classes, average="micro")]

    # Defining Callbacks
    callbacks = [
        cbs.EarlyStopping(monitor="val_accuracy", mode="max", patience=5),
        cbs.ModelCheckpoint(monitor="val_accuracy", mode="max"),
        cbs.ReduceLROnPlateau(mode="max", patience=2),
        cbs.DiscordNotifierCallback(
            exp_name="MNIST-EXP", webhook_url=os.environ.get("DISCORD_WEBHOOK")
        ),
    ]

    # Defining ModelConfig for Experiment
    config = ModelConfig(
        nn_module=Net,  # The uninstantiated model_class for the neural network.
        module_params={"out_features": 10},
        optimizer="Adam",
        optimizer_params={"lr": 3e-4},
        criterion="cross_entropy",
    )

    # Compiling and Running the experiment
    exp = Experiment(
        num_epochs=3,
        fp16=True,
        device="cuda",
        seed=42,
    )
    exp.compile_experiment(
        model_config=config,
        callbacks=callbacks,
        metrics=metric_list,
        main_metric="accuracy",
    )

    exp.fit(x=train_images, y=train_labels, val_data=(val_images, val_labels), batch_size=32)

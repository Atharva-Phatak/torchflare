# Image Classification with torchflare and Hydra

* This tutorial will guide on how to do image classification using torchflare.
* We will also use [hydra:cc](https://hydra.cc/) to manage our parameters.

Dataset: <https://www.kaggle.com/c/cifar-10>

* ### Importing Libraries
``` python

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import *

import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchflare.datasets import SimpleDataloader, show_batch
from torchflare.experiments import Experiment
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
```

* ### Loading and Preparing the dataset

``` python
df = pd.read_csv("trainLabels.csv")
classes = df.label.unique().tolist()
class_to_idx = {value: key for key, value in enumerate(classes)}
df.label = df.label.map(class_to_idx)
df.id = df.id.astype(str)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffling the dataframe

test_df = df.iloc[:10000, :]  # I took first 10000 entries as test data
data_df = df.iloc[10000:, :]
train_df, valid_df = train_test_split(
    data_df, test_size=0.3
)  # Splitting into train and validation data.
```

* ### Defining a Simple Model.
``` python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

* ### Defining basic transforms.
``` python
train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

valid_transform = transforms.Compose([transforms.ToTensor()])
```
* ### Creating Dataloaders
We will be using SimpleDataloaders from torchflare to easily create the dataloaders.
``` python
# Creating Training Dataloader.

train_dl = SimpleDataloader.image_data_from_df(
    df=train_df, augmentations=transform, **cfg.shared_data_params
).get_loader(batch_size=32, shuffle=True, num_workers=0)

# Creating Validation Dataloader.
valid_dl = SimpleDataloader.image_data_from_df(
    df=valid_df, augmentations=transform, **cfg.shared_data_params
).get_loader(batch_size=32, shuffle=False, num_workers=0)

```
* ### Defining Callbacks and metrics.

We will be using callbacks and metrics defined in torchflare library.
``` python
callbacks = [instantiate(cfg.callbacks.early_stopping),
            instantiate(cfg.callbacks.model_checkpoint)]

metrics = [instantiate(cfg.metric)]
```

* ### Setting up the Experiment
``` python
exp = Experiment(**cfg.experiment.constant_params)
exp.compile_experiment(
    model=Net(), callbacks=callbacks, metrics=metrics, **cfg.experiment.compile_params
)
exp.perform_sanity_check(train_dl)
exp.run_experiment(train_dl=train_dl, valid_dl=valid_dl)
```

* ### Running Inference
``` python
data = dict(cfg.shared_data_params)
# popping label_cols, since for test we dont need those.
_ = data.pop("label_cols")

test_dl = SimpleDataloader.image_data_from_df(
    df=test_df, augmentations=test_transform, **data
).get_loader(batch_size=32, shuffle=False)

# Inference
ops = []
for op in exp.infer(path="./models/cifar10.bin", test_loader=test_dl):
    _, y_pred = torch.max(op, dim=1)
    ops.extend(y_pred)
```
* ### Visualizing History
``` python
plot_metrics = ["loss", "accuracy"]
for metric in plot_metrics:
    exp.plot_history(key=metric, save_fig=False, plot_fig=True)
```

* ***[Notebook](/examples/image_classification_hydra.ipynb) is available in examples folder.***

![image](https://raw.githubusercontent.com/Atharva-Phatak/torchflare/main/docs/static/images/TorchFlare_official.png)

![PyPI](https://img.shields.io/pypi/v/torchflare?color=success)
![API](https://img.shields.io/badge/API-stable-success)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Atharva-Phatak/torchflare?color=success)
[![CodeFactor](https://www.codefactor.io/repository/github/atharva-phatak/torchflare/badge?s=8b602116b87a38ed9dbf6295933839ff7c85ac81)](https://www.codefactor.io/repository/github/atharva-phatak/torchflare)
[![Test](https://github.com/Atharva-Phatak/torchflare/actions/workflows/test.yml/badge.svg)](https://github.com/Atharva-Phatak/torchflare/actions/workflows/test.yml)
[![documentation](https://github.com/Atharva-Phatak/torchflare/actions/workflows/docs.yml/badge.svg)](https://github.com/Atharva-Phatak/torchflare/actions/workflows/docs.yml)
[![Publish-PyPI](https://github.com/Atharva-Phatak/torchflare/actions/workflows/publish.yml/badge.svg)](https://github.com/Atharva-Phatak/torchflare/actions/workflows/publish.yml)
[![DeepSource](https://deepsource.io/gh/Atharva-Phatak/torchflare.svg/?label=active+issues&token=_u890jqK5XjPmNlJCyQkxwmG)](https://deepsource.io/gh/Atharva-Phatak/torchflare/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/Atharva-Phatak/torchflare.svg/?label=resolved+issues&token=_u890jqK5XjPmNlJCyQkxwmG)](https://deepsource.io/gh/Atharva-Phatak/torchflare/?ref=repository-badge)
[![codecov](https://codecov.io/gh/Atharva-Phatak/torchflare/branch/main/graph/badge.svg?token=HSG3FP6NNB)](https://codecov.io/gh/Atharva-Phatak/torchflare)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Atharva-Phatak/torchflare?color=success)](https://github.com/Atharva-Phatak/torchflare/blob/main/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/torchflare?color=success)



# ***TorchFlare***

***TorchFlare*** is a simple, beginner-friendly and an easy-to-use PyTorch Framework train your models without much effort.
It provides an almost Keras-like experience for training
your models with all the callbacks, metrics, etc


### ***Features***
* _A high-level module for Keras-like training._
* _Off-the-shelf Pytorch style Datasets/Dataloaders for standard tasks such as **Image classification, Image segmentation,
  Text Classification**, etc_
* _**Callbacks** for model checkpoints, early stopping, and much more!_
* _**Metrics** and much more._
* _**Reduction** of the boiler plate code required for training your models._

![compare](https://github.com/Atharva-Phatak/torchflare/blob/main/docs/static/images/compare.gif)
***

Currently, **TorchFlare** supports ***CPU*** and ***GPU*** training. DDP and TPU support will be coming soon!

***
### ***Installation***

    pip install torchflare

***
### ***Documentation***

The Documentation is available [here](https://atharva-phatak.github.io/torchflare/)

***
### ***Stability***


The library isn't mature or stable for production use yet.


The best of the library currently would be for **non production use and rapid prototyping**.

***
### ***Getting Started***

The core idea around TorchFlare is the [Experiment](/torchflare/experiments/experiment.py)
class. It handles all the internal stuff like boiler plate code for training,
calling callbacks,metrics,etc. The only thing you need to focus on is creating you PyTorch Model.

Also, there are off-the-shelf pytorch style datasets/dataloaders available for standard tasks, so that you don't
have to worry about creating Pytorch Datasets/Dataloaders.

Here is an easy-to-understand example to show how Experiment class works.

``` python
import torch
import torch.nn as nn
from torchflare.experiments import Experiment
import torchflare.callbacks as cbs
import torchflare.metrics as metrics

#Some dummy dataloaders
train_dl = SomeTrainingDataloader()
valid_dl = SomeValidationDataloader()
test_dl = SomeTestingDataloader()
```
Create a pytorch Model

``` python
model = nn.Sequential(
    nn.Linear(num_features, hidden_state_size),
    nn.ReLU(),
    nn.Linear(hidden_state_size, num_classes)
)
```

Define callbacks and metrics
``` python
metric_list = [metrics.Accuracy(num_classes=num_classes, multilabel=False),
                metrics.F1Score(num_classes=num_classes, multilabel=False)]

callbacks = [cbs.EarlyStopping(monitor="accuracy", mode="max"), cbs.ModelCheckpoint(monitor="accuracy"),
            cbs.ReduceLROnPlateau(mode="max" , patience = 2)]
```

Define your experiment
``` python
# Set some constants for training
exp = Experiment(
    num_epochs=5,
    fp16=False,
    device="cuda",
    seed=42,
)

# Compile your experiment with model, optimizer, schedulers, etc
exp.compile_experiment(
    model = net,
    optimizer = "Adam",
    optimizer_params = {"lr" : 3e-4},
    callbacks = callbacks,
    criterion = "cross_entropy",
    metrics = metric_list,
    main_metric = "accuracy",
)

# Run your experiment with training dataloader and validation dataloader.
exp.fit_loader(train_dl=train_dl, valid_dl= valid_dl)
```

For inference, you can use infer method, which yields output per batch. You can use it as follows
``` python
outputs = []

for op in exp.predict_on_loader(test_loader=test_dl , path_to_model='./models/model.bin' , device = 'cuda'):
    op = some_post_process_function(op)
    outputs.extend(op)

```

If you want to access your experiments history or plot it. You can do it as follows.
``` python

history = exp.history # This will return a dict

# If you want to plot progress of particular metric as epoch progress use this.

exp.plot_history(keys = ["loss" , "accuracy"] , save_fig = False , plot_fig = True)
```

***
### ***Examples***
* [Image Classification](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/image_classification.ipynb) on CIFAR-10 using TorchFlare.
* [Text Classification](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/Imdb_classification.ipynb) on IMDB data.
* [Binary Classification of Tabular Data](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/tabular_classification.ipynb) on previous [kaggle competition](https://www.kaggle.com/c/cat-in-the-dat-ii/overview)
* Tutorial on using [Hydra and TorchFlare](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/image_classification_hydra.ipynb) for efficient workflow and parameter management.
* Tutorial on [fit methods](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/MNIST-Classification.ipynb) and how to dispatch training progress to your personal discord channel.
* Tutorial on how to train [Variational Autoencoders](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/MNIST-VAE.ipynb) using torchflare on MNIST Dataset.

***
### ***Current Contributors***

<a href="https://github.com/Atharva-Phatak/torchflare/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Atharva-Phatak/torchflare" />
</a>


***
### ***Contribution***


Contributions are always welcome, it would be great to have people use and contribute to this project to help users understand and benefit from the library.

#### How to contribute
- ***Create an issue:*** If you have a new feature in mind, feel free to open an issue and add some short description on what that feature could be.
- ***Create a PR***: If you have a bug fix, enhancement or new feature addition, create a Pull Request and the maintainers of the repo, would review and merge them.

***
### ***Author***

* **Atharva Phatak** - [Atharva-Phatak](https://github.com/Atharva-Phatak)

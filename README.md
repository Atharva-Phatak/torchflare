![image](https://raw.githubusercontent.com/Atharva-Phatak/torchflare/main/docs/source/_static/TorchFlare_official.png)

![PyPI](https://img.shields.io/pypi/v/torchflare?color=success)
![API](https://img.shields.io/badge/API-stable-success)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Atharva-Phatak/torchflare?color=success)
[![CodeFactor](https://www.codefactor.io/repository/github/atharva-phatak/torchflare/badge?s=8b602116b87a38ed9dbf6295933839ff7c85ac81)](https://www.codefactor.io/repository/github/atharva-phatak/torchflare)
[![Test](https://github.com/Atharva-Phatak/torchflare/actions/workflows/test.yml/badge.svg)](https://github.com/Atharva-Phatak/torchflare/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/torchflare/badge/?version=latest)](https://torchflare.readthedocs.io/en/latest/?badge=latest)
[![Publish-PyPI](https://github.com/Atharva-Phatak/torchflare/actions/workflows/publish.yml/badge.svg)](https://github.com/Atharva-Phatak/torchflare/actions/workflows/publish.yml)
[![DeepSource](https://deepsource.io/gh/Atharva-Phatak/torchflare.svg/?label=active+issues&token=_u890jqK5XjPmNlJCyQkxwmG)](https://deepsource.io/gh/Atharva-Phatak/torchflare/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/Atharva-Phatak/torchflare.svg/?label=resolved+issues&token=_u890jqK5XjPmNlJCyQkxwmG)](https://deepsource.io/gh/Atharva-Phatak/torchflare/?ref=repository-badge)
[![codecov](https://codecov.io/gh/Atharva-Phatak/torchflare/branch/main/graph/badge.svg?token=HSG3FP6NNB)](https://codecov.io/gh/Atharva-Phatak/torchflare)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Atharva-Phatak/torchflare?color=success)](https://github.com/Atharva-Phatak/torchflare/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)




# ***TorchFlare***

***TorchFlare*** is a simple, beginner-friendly and an easy-to-use PyTorch Framework to train your models with ease.
It provides an almost Keras-like experience for training
your models with all the callbacks, metrics, etc


### ***Features***
* _A high-level module for Keras-like training._
* _Flexibility to write custom training and validation loops for advanced use cases._
* _Off-the-shelf Pytorch style Datasets/Dataloaders for standard tasks such as **Image classification, Image segmentation,
  Text Classification**, etc_
* _**Callbacks** for model checkpoints, early stopping, and much more!_
* _**TorchFlare** uses powerful [torchmetrics](https://github.com/PyTorchLightning/metrics) in the backend for metric computations!_
* _**Reduction** of the boiler plate code required for training your models._
* _Create **interactive UI** for model prototyping and POC_
***

Currently, **TorchFlare** supports ***CPU*** and ***GPU*** training. DDP and TPU support will be coming soon!

***
### ***Installation***

    pip install torchflare

***
### ***Documentation***

The Documentation is available [here](https://torchflare.readthedocs.io/en/latest/)



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
import torchmetrics
import torch.nn as nn
from torchflare.experiments import Experiment, ModelConfig
import torchflare.callbacks as cbs

# Some dummy dataloaders
train_dl = SomeTrainingDataloader()
valid_dl = SomeValidationDataloader()
test_dl = SomeTestingDataloader()
```
Create a pytorch Model

``` python
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
```

Define callbacks and metrics
``` python
metric_list = [
    torchmetrics.Accuracy(num_classes=num_classes)
]

callbacks = [
    cbs.EarlyStopping(monitor="val_accuracy", mode="max"),
    cbs.ModelCheckpoint(monitor="val_accuracy"),
    cbs.ReduceLROnPlateau(mode="max", patience=2),
]
```
Define Model Configuration
``` python
#Defining Model Config for experiment.
config = ModelConfig(
    nn_module=Net,
    module_params={"n_classes": 10, "p_dropout": 0.3},
    optimizer="Adam",
    optimizer_params={"lr": 3e-4},
    criterion="cross_entropy",
)
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

exp.compile_experiment(
    model_config=config,
    callbacks=callbacks,
    metrics=metric_list,
    main_metrics="accuracy",
)
# Run your experiment with training dataloader and validation dataloader.
exp.fit_loader(train_dl=train_dl, valid_dl=valid_dl)
```

For inference, you can use infer method, which yields output per batch. You can use it as follows
``` python
outputs = []
for op in exp.predict_on_loader(
    test_loader=test_dl, path_to_model="./models/model.bin", device="cuda"
):
    op = some_post_process_function(op)
    outputs.extend(op)
```

If you want to access your experiments history or get as a dataframe. You can do it as follows.
``` python
history = exp.history  # This will return a dict
exp.get_logs()  # This will return a dataframe constructed from model-history.
```
***
### ***Examples***

* [Image Classification](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/Basic-Tutorials/image_classification.py) using TorchFlare on MNIST dataset.
* [Text Classification](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/Basic-Tutorials/text_classification.py) using Tiny Bert on IMDB dataset
* [Variational Auto-encoders](https://github.com/Atharva-Phatak/torchflare/tree/main/examples/Advanced-Tutorials/autoencoders) to generate MNIST dataset.
* Train [DCGANS](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/Advanced-Tutorials/gans/dcgan.py) to generate MNIST data.
* [Self Supervised learning](https://github.com/Atharva-Phatak/torchflare/blob/main/examples/Advanced-Tutorials/self-supervision/ssl_byol.py) using [Bootstrap your own latent(BYOL)](https://arxiv.org/abs/2006.07733)

***
### ***Contributions***
To contribute please refer to [Contributing Guide](https://github.com/Atharva-Phatak/torchflare/blob/main/.github/CONTRIBUTING.MD)

***
### ***Current Contributors***

<a href="https://github.com/Atharva-Phatak/torchflare/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Atharva-Phatak/torchflare" />
</a>

***

### ***Author***

* **Atharva Phatak** - [Atharva-Phatak](https://github.com/Atharva-Phatak)


### Citation

Please use this bibtex if you want to cite this repository in your publications:

    @misc{TorchFlare,
        author = {Atharva Phatak},
        title = {TorchFlare: Easy model training and experimentation.},
        year = {2020},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/Atharva-Phatak/torchflare}},
    }

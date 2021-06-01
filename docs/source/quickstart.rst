Quick Start
==============


Simple Example - MNIST Dataset
-----------------------------------

1. Define Pytorch Model

.. code-block:: python

    import torch
    from torch import nn
    import torch.nn.functional as F

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

2. Download MNIST dataset. Create validation and training PyTorch data loaders.

.. code-block:: python

    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.datasets import MNIST

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_mnist_dataset = MNIST(download=True, root="mnist_data",
                                transform=data_transform, train=True)
    val_mnist_dataset = MNIST(download=False, root="mnist_data",
                              transform=data_transform, train=False)
    train_dl = DataLoader(train_mnist_dataset,
                              batch_size=64, shuffle=True)
    val_dl = DataLoader(val_mnist_dataset,
                            batch_size=128, shuffle=False)

3. Define Callbacks and Metrics

.. code-block:: python

    import torchflare.callbacks as cbs
    import torchflare.metrics as metrics

    metric_list = [metrics.Accuracy(num_classes = 10, multilabel = False)]

    callbacks = [cbs.EarlyStoppingCallback(monitor = "accuracy" , mode = "max"),
                cbs.ModelCheckpointCallback(monitor = "accuracy", mode = "max"),
                cbs.ReduceLROnPlateau(monitor='accuracy', patience=3)]

4. Define and Compile the Experiment

.. code-block:: python

    from torchflare.experiments import Experiment

    # Define some params for the experiment
    exp = Experiment(num_epochs=10,
                fp16=True,
                device="cuda",
                seed=42)

    # Compile the experiment
    exp.compile_experiment(module = Net,
                          module_params = {"n_classes" : 10 , "p_dropout" : 0.3},
                          optimizer = "Adam"
                          optimizer_params = {"lr" : 3e-4},
                          criterion = "cross_entropy",
                          callbacks = callbacks,
                          metrics = metric_list,
                          main_metrics = "accuracy")

    #Run the experiment
    exp.fit_loader(train_dl = train_dl , valid_dl)

    # Get logs for the experiment

    logs = exp.get_logs()

More examples are available in `Github repo <https://github.com/Atharva-Phatak/torchflare/tree/main/examples>`_.

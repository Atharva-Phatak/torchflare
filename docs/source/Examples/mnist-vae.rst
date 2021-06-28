Data generation using Variational Autoencoders
=====================================================

1. Import the necessary libraries

.. code-block:: python

    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from matplotlib import pyplot as plt
    import torch.nn.functional as F
    from torchflare.experiments import Experiment, ModelConfig

2. Create dataloaders

.. code-block:: python

    train_dataset = MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    bs = 64
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=bs, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=bs, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=bs, shuffle=False
    )

3. Define the model

.. code-block:: python

    class VAE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.encoder = nn.Sequential(
                nn.Linear(784, self.d ** 2), nn.ReLU(), nn.Linear(self.d ** 2, self.d * 2)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.d, self.d ** 2),
                nn.ReLU(),
                nn.Linear(self.d ** 2, 784),
                nn.Sigmoid(),
            )

        def reparameterise(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()
                eps = std.data.new(std.size()).normal_()
                return eps.mul(std).add_(mu)
            else:
                return mu

        def forward(self, x):
            mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, self.d)
            mu = mu_logvar[:, 0, :]
            logvar = mu_logvar[:, 1, :]
            z = self.reparameterise(mu, logvar)
            return self.decoder(z), mu, logvar

4. Define the loss function

.. code-block:: python

    def loss_function(x_hat, x, mu, logvar, β=1):
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x.view(-1, 784), reduction='sum'
        )
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        return BCE + β * KLD

5. Define custom Experiment for Training

    Here we override ``batch_step`` method in Experiment class.
    Remember when you override ``batch_step`` you have to mandatorily assign values to ``self.preds``, ``self.loss`` and ``self.loss_per_batch``

    .. code-block:: python

        class VAEExperiment(Experiment):
            def batch_step(self):
                self.preds = self.state.model(self.batch[self.input_key])
                x_hat, mu, logvar = self.preds
                self.loss = self.state.criterion(x_hat, self.batch[self.input_key], mu, logvar)
                self.loss_per_batch = {"loss": self.loss.item()}


6. Define model config and run the experiment

.. code-block:: python


    config = ModelConfig(
        nn_module=VAE,
        module_params={"d": 20},
        optimizer="Adam",
        optimizer_params={"lr": 3e-3},
        criterion=loss_function,
    )


    exp = VAEExperiment(num_epochs=30, fp16=False, device="cuda", seed=42)
    exp.compile_experiment(model_config=config)
    exp.fit_loader(train_loader, val_loader)


More examples are available in `Github repo <https://github.com/Atharva-Phatak/torchflare/tree/main/examples>`_.

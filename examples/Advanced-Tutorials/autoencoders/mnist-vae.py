"""Generating MNIST digits using Variational Autoencoders."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from torchflare.experiments import Experiment, ModelConfig


# Defining VAE module.
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


# Defining the loss functions.
def loss_function(preds, beta=1):
    x_hat, x, mu, logvar = preds
    BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction="sum")
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE + beta * KLD


if __name__ == "__main___":
    # Creating dataloaders.
    train_dataset = MNIST(
        root="./mnist_data/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = MNIST(
        root="./mnist_data/", train=False, transform=transforms.ToTensor(), download=False
    )

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    bs = 64
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # Defining config for experiment.
    config = ModelConfig(
        nn_module=VAE,
        module_params={"d": 20},
        optimizer="Adam",
        optimizer_params={"lr": 3e-3},
        criterion=loss_function,
    )

    # Compiling and running experiment.
    exp = Experiment(num_epochs=30, fp16=False, device="cuda", seed=42)
    exp.compile_experiment(model_config=config)
    exp.fit_loader(train_loader, val_loader)

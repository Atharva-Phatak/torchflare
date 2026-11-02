"""Implementation of Bootstrap your own latent: A new approach to self-supervised Learning
Paper : https://arxiv.org/abs/2006.07733
"""

import random
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision
from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as ktf
from torch import Tensor, nn
from torchvision.transforms import ToTensor

import torchflare.callbacks as cbs
from torchflare.experiments import Experiment, ModelConfig


# Defining custom callback using callback decorators.
@cbs.on_experiment_start(order=cbs.CallbackOrder.MODEL_INIT)
def init_target_network(experiment: "Experiment"):
    for online_params, target_params in zip(
        experiment.state.model["online_network"],
        experiment.state.model["target_network"],
    ):
        target_params.data.copy_(online_params.data)
        target_params.requires_grad = False


# Defining the loss function
def normalized_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.mean(2 - 2 * (x * y).sum(dim=-1))


# Defining augmentations using kornia
class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def default_augmentation(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    return nn.Sequential(
        ktf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )


# Defining the models.
class MLPHead(nn.Module):
    def __init__(self, in_channels: int, projection_size: int = 256, hidden_size: int = 4096):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


# Defining resnet encoders.
class ResnetEncoder(nn.Module):
    def __init__(self, pretrained, mlp_params):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projector = MLPHead(in_channels=resnet.fc.in_features, **mlp_params)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projector(h)


# Defining custom training method required as required by Bootstrap your own latent.(SSL)
class BYOLExperiment(Experiment):
    def __init__(self, momentum, augmentation_fn, image_size, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.augmentation_fn = augmentation_fn(image_size)

    def get_model_params(self, config):
        if config.model_dict and not config.optimizer_dict:
            grad_params = list(self.state.model["online_network"].parameters()) + list(
                self.state.model["predictor"].parameters()
            )
            return grad_params

    @torch.no_grad()
    def update_target_network(self):
        for online_params, target_params in zip(
            self.state.model["online_network"].parameters(),
            self.state.model["target_network"].parameters(),
        ):
            target_params.data = (
                target_params.data * self.momentum + online_params.data * self.momentum
            )

    def train_step(self) -> Dict:

        self.backend.zero_grad(optimizer=self.state.optimizer)
        x = self.batch[self.input_key]
        view_1, view_2 = self.augmentation_fn(x), self.augmentation_fn(x)
        pred_1 = self.state.model["predictor"](self.state.model["online_network"](view_1))
        pred_2 = self.state.model["predictor"](self.state.model["online_network"](view_2))

        with torch.no_grad():
            target_2 = self.state.model["target_network"](view_1)
            target_1 = self.state.model["target_network"](view_2)

        loss = self.state.criterion(pred_1, target_1) + self.state.criterion(pred_2, target_2)
        self.backend.backward_loss(loss=loss)
        self.backend.optimizer_step(optimizer=self.state.optimizer)
        self.update_target_network()
        return {self.loss_key: loss}


if __name__ == "__main__":
    # Creating training dataloader
    train_data = torchvision.datasets.STL10(
        root="./", split="unlabeled", transform=ToTensor(), download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=2048,
        shuffle=True,
    )

    # Defining config for experiment.
    config = ModelConfig(
        nn_module={
            "online_network": ResnetEncoder,
            "target_network": ResnetEncoder,
            "predictor": MLPHead,
        },
        module_params={
            "online_network": {
                "pretrained": True,
                "mlp_params": {"projection_size": 256, "hidden_size": 1024},
            },
            "target_network": {
                "pretrained": True,
                "mlp_params": {"projection_size": 256, "hidden_size": 1024},
            },
            "predictor": {
                "in_channels": 256,
                "projection_size": 256,
                "hidden_size": 1024,
            },
        },
        optimizer="AdamW",
        optimizer_params={"lr": 1e-4, "weight_decay": 1e-3},
        criterion=normalized_mse,
    )

    # Defining callbacks.
    callbacks = [
        cbs.ModelCheckpoint(save_dir="./", mode="min", monitor="train_loss"),
        init_target_network,
        cbs.CosineAnnealingWarmRestarts(T_0=2),
    ]

    # Compiling and running the experiment.
    byol_exp = BYOLExperiment(
        num_epochs=25,
        seed=42,
        device="cuda",
        fp16=True,
        momentum=4e-3,
        augmentation_fn=default_augmentation,
        image_size=(96, 96),
    )

    byol_exp.compile_experiment(model_config=config, callbacks=callbacks, metrics=None)
    byol_exp.fit_loader(train_loader)

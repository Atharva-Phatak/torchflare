Training DCGANs
============================

1. Import the necessary libraries

.. code-block:: python

    import torch
    from torch import nn
    from torchflare.experiments import Experiment, ModelConfig
    import torchflare.callbacks as cbs
    import torchvision as tv
    from torchvision.datasets import ImageFolder
    import os

2. Define the architecture for the Generator

.. code-block:: python

    class Generator(nn.Module):
        def __init__(self, latent_dim, batchnorm=True):
            """A generator for mapping a latent space to a sample space.
            The sample space for this generator is single-channel, 28x28 images
            with pixel intensity ranging from -1 to +1.
            Args:
                latent_dim (int): latent dimension ("noise vector")
                batchnorm (bool): Whether or not to use batch normalization
            """
            super(Generator, self).__init__()
            self.latent_dim = latent_dim
            self.batchnorm = batchnorm
            self._init_modules()

        def _init_modules(self):
            """Initialize the modules."""
            # Project the input
            self.linear1 = nn.Linear(self.latent_dim, 256 * 7 * 7, bias=False)
            self.bn1d1 = nn.BatchNorm1d(256 * 7 * 7) if self.batchnorm else None
            self.leaky_relu = nn.LeakyReLU()

            # Convolutions
            self.conv1 = nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            )
            self.bn2d1 = nn.BatchNorm2d(128) if self.batchnorm else None

            self.conv2 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            self.bn2d2 = nn.BatchNorm2d(64) if self.batchnorm else None

            self.conv3 = nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            self.tanh = nn.Tanh()

        def forward(self, input_tensor):
            """Forward pass; map latent vectors to samples."""
            intermediate = self.linear1(input_tensor)
            intermediate = self.bn1d1(intermediate)
            intermediate = self.leaky_relu(intermediate)

            intermediate = intermediate.view((-1, 256, 7, 7))

            intermediate = self.conv1(intermediate)
            if self.batchnorm:
                intermediate = self.bn2d1(intermediate)
            intermediate = self.leaky_relu(intermediate)

            intermediate = self.conv2(intermediate)
            if self.batchnorm:
                intermediate = self.bn2d2(intermediate)
            intermediate = self.leaky_relu(intermediate)

            intermediate = self.conv3(intermediate)
            output_tensor = self.tanh(intermediate)
            return output_tensor

3. Define the architecture for the Discriminator.

.. code-block:: python

    class Discriminator(nn.Module):
        def __init__(self, output_dim):
            """A discriminator for discerning real from generated images.
            Images must be single-channel and 28x28 pixels.
            Output activation is Sigmoid.
            """
            super(Discriminator, self).__init__()
            self.output_dim = output_dim
            self._init_modules()  # I know this is overly-organized. Fight me.

        def _init_modules(self):
            """Initialize the modules."""
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True,
            )
            self.leaky_relu = nn.LeakyReLU()
            self.dropout_2d = nn.Dropout2d(0.3)

            self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True,
            )

            self.linear1 = nn.Linear(128 * 7 * 7, self.output_dim, bias=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_tensor):
            """Forward pass; map samples to confidence they are real [0, 1]."""
            intermediate = self.conv1(input_tensor)
            intermediate = self.leaky_relu(intermediate)
            intermediate = self.dropout_2d(intermediate)

            intermediate = self.conv2(intermediate)
            intermediate = self.leaky_relu(intermediate)
            intermediate = self.dropout_2d(intermediate)

            intermediate = intermediate.view((-1, 128 * 7 * 7))
            intermediate = self.linear1(intermediate)
            output_tensor = self.sigmoid(intermediate)

            return output_tensor


3. Define a custom train step.

    To define a custom train step in TorchFlare you need to wrapped inheriting ``Experiment`` class and override the method named ``train_step``.
    However you have to keep following things in mind when you override the method.

    a) Train step in TorchFlare involves forward pass from the model, loss computation, backward pass and optimizer step.
    b) Train step must return a dictionary with atleast loss value.

    .. code-block:: python



        # Defining Custom Loop for training
        class DCGANExperiment(Experiment):
            def __init__(self, latent_dim, batch_size, **kwargs):

                super(DCGANExperiment, self).__init__(**kwargs)

                self.noise_fn = lambda x: torch.randn((x, latent_dim), device=self.device)
                self.target_ones = torch.ones((batch_size, 1), device=self.device)
                self.target_zeros = torch.zeros((batch_size, 1), device=self.device)

            def train_step(self):

                latent_vec = self.noise_fn(self.batch[self.input_key].shape[0])

                # self.backend has methods like zero_grad, etc to handle the backward pass, optimizer_step and zero_grad.
                self.backend.zero_grad(self.state.optimizer["discriminator"])
                pred_real = self.state.model["discriminator"](self.batch[self.input_key])

                loss_real = self.state.criterion(pred_real, self.target_ones)

                fake = self.state.model["generator"](latent_vec)
                pred_fake = self.state.model["discriminator"](fake.detach())
                loss_fake = self.state.criterion(pred_fake, self.target_zeros)

                loss_d = (loss_real + loss_fake) / 2
                self.backend.backward_loss(loss_d)

                self.backend.optimizer_step(self.state.optimizer["discriminator"])

                # Generator Training

                self.backend.zero_grad(self.state.optimizer["generator"])
                classifications = self.state.model["discriminator"](fake)
                loss_g = self.state.criterion(classifications, self.target_ones)
                self.backend.backward_loss(loss_g)
                self.backend.optimizer_step(self.state.optimizer["generator"])

                return {"loss_g": loss_g.item(), "loss_d": loss_d.item()}

4. Create dataloaders

.. code-block:: python

    batch_size = 16
    latent_dim = 16
    transform = tv.transforms.Compose(
        [
            tv.transforms.Grayscale(num_output_channels=1),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = ImageFolder(root=os.path.join("mnist_png", "training"), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

5. Define the model configs and callbacks.

.. code-block:: python

    config = ModelConfig(
        nn_module={"discriminator": Discriminator, "generator": Generator},
        module_params={
            "discriminator": {"output_dim": 1},
            "generator": {"latent_dim": latent_dim},
        },
        optimizer={"discriminator": "Adam", "generator": "Adam"},
        optimizer_params={"discriminator": dict(lr=1e-3), "generator": dict(lr=2e-4)},
        criterion="binary_cross_entropy",
    )

    callbacks = [cbs.ModelCheckpoint(mode="min", monitor="train_loss_g", save_dir="./")]

6. Compile and Run the experiment.

.. code-block:: python


    exp = DCGANExperiment(
        latent_dim=latent_dim,
        batch_size=batch_size,
        num_epochs=1,
        device="cuda",
        seed=42,
        fp16=False,
    )

    exp.compile_experiment(model_config=exp_config, callbacks=callbacks)
    exp.fit_loader(dataloader)

More examples are available in `Github repo <https://github.com/Atharva-Phatak/torchflare/tree/main/examples>`_.

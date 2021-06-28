Callbacks
============================
.. toctree::
   :titlesonly:

.. contents::
   :local:


Early Stopping
-------------------------------------------

.. autoclass:: torchflare.callbacks.EarlyStopping
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Model Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.ModelCheckpoint
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Load Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.LoadCheckpoint
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Learning Rate Scheduler Callbacks
---------------------------------------------
**Callbacks for auto adjust the learning rate based on the number of epochs or other metrics measurements.**
These callbacks are wrappers of native PyTorch :mod:`torch.optim.lr_scheduler`.

StepLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.StepLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

LambdaLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.LambdaLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

MultiStepLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.MultiStepLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

ExponentialLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.ExponentialLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

CosineAnnealingLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.CosineAnnealingLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

ReduceLROnPlateau
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.ReduceLROnPlateau
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

CyclicLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.CyclicLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

CosineAnnealingWarmRestarts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.CosineAnnealingWarmRestarts
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

MultiplicativeLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.MultiplicativeLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

OneCycleLR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.OneCycleLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging Callbacks
----------------------------------

**Logging module integrates various logging services like neptune, comet, etc for logging the progress of experiments.**

Comet Logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.CometLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Neptune Logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.NeptuneLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Wandb Logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.WandbLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Tensorboard Logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.callbacks.TensorboardLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Message Notifiers
----------------------------
**Message notifiers send training progress per epoch to your personal slack and discord channels.**

Discord Notifier Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.callbacks.DiscordNotifierCallback
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Slack Notifier Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.callbacks.SlackNotifierCallback
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Decorators
------------------------------------

The main callback decorators simply take function , order as inputs and bind it to a callback point, returning the result.

.. autofunction:: torchflare.callbacks.on_experiment_start
.. autofunction:: torchflare.callbacks.on_epoch_start
.. autofunction:: torchflare.callbacks.on_loader_start
.. autofunction:: torchflare.callbacks.on_batch_start
.. autofunction:: torchflare.callbacks.on_experiment_end
.. autofunction:: torchflare.callbacks.on_epoch_end
.. autofunction:: torchflare.callbacks.on_loader_end
.. autofunction:: torchflare.callbacks.on_batch_end

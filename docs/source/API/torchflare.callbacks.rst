Callbacks
============================
.. toctree::
   :titlesonly:

.. contents::
   :local:


Early Stopping
-------------------------------------------

.. autoclass:: torchflare.callbacks.early_stopping.EarlyStopping
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Model Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.model_checkpoint.ModelCheckpoint
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Load Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.load_checkpoint.LoadCheckpoint
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - StepLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.StepLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - LambdaLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.LambdaLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - MultiStepLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.MultiStepLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - ExponentialLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.ExponentialLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CosineAnnealingLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CosineAnnealingLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - ReduceLROnPlateau
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.ReduceLROnPlateau
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CyclicLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CyclicLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CosineAnnealingWarmRestarts
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CosineAnnealingWarmRestarts
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - MultiplicativeLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.MultiplicativeLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - OneCycleLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.OneCycleLR
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end


Logging - Comet Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.comet_logger.CometLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Neptune Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.neptune_logger.NeptuneLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Wandb Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.wandb_logger.WandbLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Tensorboard Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.tensorboard_logger.TensorboardLogger
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end


Notification - Discord Notifier Callback
----------------------------------------------------
.. autoclass:: torchflare.callbacks.message_notifiers.DiscordNotifierCallback
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Notification - Slack Notifier Callback
-------------------------------------------------
.. autoclass:: torchflare.callbacks.message_notifiers.SlackNotifierCallback
   :members:
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Decorators
------------------------------------

Main
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The main callback decorators simply take function , order as inputs and bind it to a callback point, returning the result.

.. autofunction:: torchflare.callbacks.callback_decorators.on_experiment_start
.. autofunction:: torchflare.callbacks.callback_decorators.on_epoch_start
.. autofunction:: torchflare.callbacks.callback_decorators.on_loader_start
.. autofunction:: torchflare.callbacks.callback_decorators.on_batch_start
.. autofunction:: torchflare.callbacks.callback_decorators.on_experiment_end
.. autofunction:: torchflare.callbacks.callback_decorators.on_epoch_end
.. autofunction:: torchflare.callbacks.callback_decorators.on_loader_end
.. autofunction:: torchflare.callbacks.callback_decorators.on_batch_end

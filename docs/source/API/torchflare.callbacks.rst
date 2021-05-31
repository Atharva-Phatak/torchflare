Callbacks
============================
.. toctree::
   :titlesonly:

.. contents::
   :local:


Early Stopping
-------------------------------------------

.. autoclass:: torchflare.callbacks.early_stopping.EarlyStopping
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Model Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.model_checkpoint.ModelCheckpoint
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Load Checkpoint
-------------------------------------------

.. autoclass:: torchflare.callbacks.load_checkpoint.LoadCheckpoint
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - StepLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.StepLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - LambdaLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.LambdaLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - MultiStepLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.MultiStepLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - ExponentialLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.ExponentialLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CosineAnnealingLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CosineAnnealingLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - ReduceLROnPlateau
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.ReduceLROnPlateau
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CyclicLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CyclicLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - CosineAnnealingWarmRestarts
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.CosineAnnealingWarmRestarts
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - MultiplicativeLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.MultiplicativeLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Scheduler - OneCycleLR
------------------------------------------

.. autoclass:: torchflare.callbacks.lr_schedulers.OneCycleLR
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end


Logging - Comet Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.comet_logger.CometLogger
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Neptune Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.neptune_logger.NeptuneLogger
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Wandb Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.wandb_logger.WandbLogger
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Logging - Tensorboard Logger
-----------------------------------------

.. autoclass:: torchflare.callbacks.tensorboard_logger.TensorboardLogger
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end


Notification - Discord Notifier Callback
----------------------------------------------------
.. automodule:: torchflare.callbacks.message_notifiers.DiscordNotifierCallback
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

Notification - Slack Notifier Callback
-------------------------------------------------
.. automodule:: torchflare.callbacks.message_notifiers.SlackNotifierCallback
   :members: __init__
   :exclude-members: on_experiment_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end, on_experiment_end

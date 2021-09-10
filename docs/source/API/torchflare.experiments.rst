Experiments
==============================

Experiment
---------------------------
**Experiment class handles all the internal stuff like boiler plate code for training, calling callbacks,metrics,etc.**
**One can override some ``train_step``, ``val_step`` and ``batch_step`` in experiment class perform custom training and validation.**

.. autoclass:: torchflare.experiments.Experiment
   :members: compile_experiment, fit ,fit_loader, predict, predict_on_loader, train_step, val_step
   :exclude-members:  on_experiment_start, on_experiment_end,on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end,
                   run_batch,run_loader,set_dataloaders,

ModelConfig
--------------------------
.. autoclass:: torchflare.experiments.ModelConfig

Backends
--------------------------

BaseBackend
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.experiments.BaseBackend
   :members: zero_grad, backward_loss, optimizer_step

AMPBackend
^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.experiments.AMPBackend
   :members: zero_grad, backward_loss, optimizer_step

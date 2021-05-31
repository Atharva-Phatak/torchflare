Experiment
==============================

.. autoclass:: torchflare.experiments.experiment.Experiment
   :members: compile_experiment, fit ,fit_loader, predict, predict_on_loader
   :exclude-members:  on_experiment_start, on_experiment_end,on_stage_start, on_epoch_start, on_loader_start, on_batch_start, on_batch_end, on_loader_end, on_epoch_end,
                compute_loss, model_forward_pass, train_step, val_step, process_inputs, run_batch,run_loader,set_dataloaders,

Example
--------------------
.. code-block:: python

    import torch
    import torchflare.callbacks as cbs
    import torchflare.metrics as metrics
    from torchflare.experiments import Experiment

    # Defining Training/Validation Dataloaders
    train_dl = SomeTrainDataloader()
    valid_dl = SomeValidDataloader()

    # Defining params
    optimizer = "Adam"
    optimizer_params = {"lr" : 1e-4}
    criterion = "cross_entropy"
    num_epochs = 10
    num_classes = 4

    # Defining the list of metrics
    metric_list = [
        metrics.Accuracy(num_classes=num_classes, multilabel=False),
        metrics.F1Score(num_classes=num_classes, multilabel=False),
    ]

    # Defining the list of callbacks
    callbacks = [
        cbs.EarlyStopping(monitor="accuracy", mode="max"),
        cbs.ModelCheckpoint(monitor="accuracy", mode = "max"),
        cbs.ReduceLROnPlateau(mode = "max" , patience = 3) #Defining Scheduler callback.
    ]

    # Creating Experiment and setting the params.
    exp = Experiment(
        num_epochs=num_epochs,
        fp16=True,
        device=device,
        seed=42,
    )

    # Compiling the experiment
    exp.compile_experiment(
        module=SomeModelClass,
        module_params = {"num_features" : 200 , "num_classes" : 5} #Params to init the model class
        metrics=metric_list,
        callbacks=callbacks,
        main_metric="accuracy",
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        criterion=criterion,
    )


    # Running the experiment
    exp.fit_loader(train_dl=train_dl, valid_dl=valid_dl)

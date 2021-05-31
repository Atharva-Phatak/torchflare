.. TorchFlare documentation master file, created by
   sphinx-quickstart on Sat May 29 15:33:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchFlare
======================================
**TorchFlare** is a simple, beginner-friendly and an easy-to-use PyTorch Framework train your models without much effort. It provides an almost Keras-like experience for training your models with all the callbacks, metrics, etc

**Features**

    * A high-level module for Keras-like training.
    * Off-the-shelf Pytorch style Datasets/Dataloaders for standard tasks such as Image classification, Image segmentation, Text Classification, etc
    * Callbacks for model checkpoints, early stopping, and much more!
    * Metrics and much more.
    * Reduction of the boiler plate code required for training your models.

Currently, TorchFlare supports CPU and GPU training. DDP and TPU support will be coming soon!

*Installation*
------------------
pip install torchflare

*Getting Started*
------------------

The core idea around TorchFlare is the Experiment class. It handles all the internal stuff like boiler plate code for training, calling callbacks,metrics,etc. The only thing you need to focus on is creating you PyTorch Model.

Also, there are off-the-shelf pytorch style datasets/dataloaders available for standard tasks, so that you don't have to worry about creating Pytorch Datasets/Dataloaders.

Here is an easy-to-understand example to show how Experiment class works.

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torchflare.experiment import Experiment
   import torchflare.callbacks as cbs
   import torchflare.metrics as metrics

   #Some dummy dataloaders
   train_dl = SomeTrainingDataloader()
   valid_dl = SomeValidationDataloader()
   test_dl = SomeTestingDataloader()

   metric_list = [metrics.Accuracy(num_classes=num_classes, multilabel=False),
                metrics.F1Score(num_classes=num_classes, multilabel=False)]

   callbacks = [cbs.EarlyStopping(monitor="accuracy", mode="max"), cbs.ModelCheckpoint(monitor="accuracy"),
            cbs.ReduceLROnPlateau(mode="max" , patience = 2)]

      # Set some constants for training
   exp = Experiment(
       num_epochs=5,
       fp16=False,
       device="cuda",
       seed=42,
   )

   # Compile your experiment with model, optimizer, schedulers, etc
   exp.compile_experiment(
       module = ModelClass,
       module_params = {"in_features" : 200 , "num_classes" : 5}
       optimizer = "Adam",
       optimizer_params = {"lr" : 3e-4},
       callbacks = callbacks,
       criterion = "cross_entropy",
       metrics = metric_list,
       main_metric = "accuracy",
   )

   # Run your experiment with training dataloader and validation dataloader.
   exp.fit_loader(train_dl=train_dl, valid_dl= valid_dl)

   #Do Inference
   outputs = []

   for op in exp.predict_on_loader(test_loader=test_dl , path_to_model='./models/model.bin' , device = 'cuda'):
       op = some_post_process_function(op)
       outputs.extend(op)

   #Visualize Model History
   exp.plot_history(keys = ["loss" , "accuracy"] , save_fig = False , plot_fig = True)



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: API

   API/torchflare.experiments.rst
   API/torchflare.callbacks.rst
   API/torchflare.datasets.rst
   API/torchflare.metrics.rst
   API/torchflare.criterion.rst
   API/torchflare.modules.rst
   API/torchflare.interpreters.rst

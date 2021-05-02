::: torchflare.experiments.experiment.Experiment
    handler: python
    selection:
      members:
        - __init__
        - compile_experiment
        - run_experiment
        - infer
        - perform_sanity_check
    rendering:
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false

## Examples

``` python
import torch
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
from torchflare.experiments import Experiment

# Defining Training/Validation Dataloaders
train_dl = SomeTrainDataloader()
valid_dl = SomeValidDataloader()

# Defining some basic model
model = SomeModel()

# Defining params
optimizer = "Adam"
optimizer_params = dict(lr=1e-4)
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
    save_dir="./test_save",
    model_name="test_classification.bin",
    fp16=True,
    device=device,
    seed=42,
    compute_train_metrics=False,
)

# Compiling the experiment
exp.compile_experiment(
    model=model,
    metrics=metric_list,
    callbacks=callbacks,
    main_metric="accuracy",
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    criterion=criterion,
)

# Performing sanity check(optional)
exp.perform_sanity_check(train_dl)

# Running the experiment
exp.run_experiment(train_dl=train_dl, valid_dl=valid_dl)
```

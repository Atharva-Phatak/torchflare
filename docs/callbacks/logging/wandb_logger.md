::: torchflare.callbacks.logging.wandb_logger.WandbLogger
    rendering:
         show_root_toc_entry: false

## Examples

``` python
from torchflare.callbacks import WandbLogger

params = {"bs": 16, "lr": 0.3}

logger = WandbLogger(
    project="Experiment",
    entity="username",
    name="Experiment_10",
    config=params,
    tags=["Experiment", "fold_0"],
)
```

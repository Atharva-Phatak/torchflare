::: torchflare.callbacks.logging.neptune_logger.NeptuneLogger
    rendering:
         show_root_toc_entry: false


## Examples

``` python
from torchflare.callbacks import NeptuneLogger

params = {"bs": 16, "lr": 0.3}

logger = NeptuneLogger(
    project_dir="username/Experiments",
    params=params,
    experiment_name="Experiment_10",
    tags=["Experiment", "fold_0"],
    api_token="your_secret_api_token",
)
```

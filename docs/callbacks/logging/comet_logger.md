::: torchflare.callbacks.logging.comet_logger.CometLogger
    rendering:
         show_root_toc_entry: false

## Examples

``` python
from torchflare.callbacks import CometLogger

params = {"bs": 16, "lr": 0.3}

logger = CometLogger(
    project_name="experiment_10",
    workspace="username",
    params=params,
    tags=["Experiment", "fold_0"],
    api_token="your_secret_api_token",
)
```

::: torchflare.callbacks.early_stopping.EarlyStopping

## Examples

``` python
import torchflare.callbacks as cbs

early_stop = cbs.EarlyStopping(monitor="val_accuracy", patience=5, mode="max")
```

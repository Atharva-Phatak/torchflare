::: torchflare.callbacks.early_stopping.EarlyStopping
    rendering:
         show_root_toc_entry: false

## Examples

``` python
import torchflare.callbacks as cbs

early_stop = cbs.EarlyStopping(monitor="val_accuracy", patience=5, mode="max")
```

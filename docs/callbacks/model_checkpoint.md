::: torchflare.callbacks.model_checkpoint.ModelCheckpoint
    rendering:
         show_root_toc_entry: false

## Examples

``` python
import torchflare.callbacks as cbs

model_ckpt = cbs.ModelCheckpoint(monitor="val_accuracy", mode="max")
```

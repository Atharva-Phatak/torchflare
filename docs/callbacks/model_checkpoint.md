::: torchflare.callbacks.model_checkpoint.ModelCheckpoint

## Examples

``` python
import torchflare.callbacks as cbs

model_ckpt = cbs.ModelCheckpoint(monitor="val_accuracy", mode="max")
```

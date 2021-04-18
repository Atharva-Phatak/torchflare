::: torchflare.metrics.fbeta_meter.FBeta
    rendering:
         show_root_toc_entry: false

## Examples

``` python
from torchflare.metrics import FBeta

# Binary-Classification Problems
acc = FBeta(num_classes=2, threshold=0.7, multilabel=False, average="macro")

# Mutliclass-Classification Problems
multiclass_acc = FBeta(num_classes=4, multilabel=False, average="macro")

# Multilabel-Classification Problems
multilabel_acc = FBeta(num_classes=5, multilabel=True, threshold=0.7, average="macro")
```

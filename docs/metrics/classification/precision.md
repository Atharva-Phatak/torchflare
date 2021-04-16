::: torchflare.metrics.precision_meter.Precision

## Examples

``` python
from torchflare.metrics import Precision

# Binary-Classification Problems
acc = Precision(num_classes=1, threshold=0.7, multilabel=False, average="macro")

# Mutliclass-Classification Problems
multiclass_acc = Precision(num_classes=4, multilabel=False, average="macro")

# Multilabel-Classification Problems
multilabel_acc = Precision(
    num_classes=5, multilabel=True, threshold=0.7, average="macro"
)
```

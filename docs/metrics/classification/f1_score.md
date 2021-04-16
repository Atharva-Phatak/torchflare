::: torchflare.metrics.fbeta_meter.F1Score

## Examples

``` python
from torchflare.metrics import F1Score

# Binary-Classification Problems
acc = F1Score(num_classes=1, threshold=0.7, multilabel=False, average="macro")

# Mutliclass-Classification Problems
multiclass_acc = F1Score(num_classes=4, multilabel=False, average="macro")

# Multilabel-Classification Problems
multilabel_acc = F1Score(num_classes=5, multilabel=True, threshold=0.7, average="macro")
```

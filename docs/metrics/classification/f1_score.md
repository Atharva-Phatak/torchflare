::: torchflare.metrics.fbeta_meter.F1Score
    rendering:
         show_root_toc_entry: false

## Examples

``` python
from torchflare.metrics import F1Score

# Binary-Classification Problems
acc = F1Score(num_classes=2, threshold=0.7, multilabel=False, average="macro")

# Mutliclass-Classification Problems
multiclass_acc = F1Score(num_classes=4, multilabel=False, average="macro")

# Multilabel-Classification Problems
multilabel_acc = F1Score(num_classes=5, multilabel=True, threshold=0.7, average="macro")
```

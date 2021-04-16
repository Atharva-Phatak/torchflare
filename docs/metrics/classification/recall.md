::: torchflare.metrics.recall_meter.Recall

## Examples

``` python

from torchflare.metrics import Recall

# Binary-Classification Problems
acc = Recall(num_classes=1 , threshold=0.7 , multilabel=False , average = "macro")

# Mutliclass-Classification Problems
multiclass_acc = Recall(num_classes=4 , multilabel=False , average = "macro")

# Multilabel-Classification Problems
multilabel_acc = Recallnum_classes=5 , multilabel=True, threshold=0.7,
                            average = "macro")
```

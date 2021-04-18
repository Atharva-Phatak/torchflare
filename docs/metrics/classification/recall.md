::: torchflare.metrics.recall_meter.Recall
    rendering:
         show_root_toc_entry: false

## Examples

``` python

from torchflare.metrics import Recall

# Binary-Classification Problems
acc = Recall(num_classes=2 , threshold=0.7 , multilabel=False , average = "macro")

# Mutliclass-Classification Problems
multiclass_acc = Recall(num_classes=4 , multilabel=False , average = "macro")

# Multilabel-Classification Problems
multilabel_acc = Recallnum_classes=5 , multilabel=True, threshold=0.7,
                            average = "macro")
```

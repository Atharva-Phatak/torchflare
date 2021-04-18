::: torchflare.metrics.accuracy_meter.Accuracy
    rendering:
         show_root_toc_entry: false

## Examples

```python
from torchflare.metrics import Accuracy

# Binary-Classification Problems
acc = Accuracy(num_classes=2, threshold=0.7, multilabel=False)

# Mutliclass-Classification Problems
multiclass_acc = Accuracy(num_classes=4, multilabel=False)

# Multilabel-Classification Problems
multilabel_acc = Accuracy(num_classes=5, multilabel=True, threshold=0.7)
```

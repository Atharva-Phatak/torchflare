::: torchflare.metrics.accuracy_meter.Accuracy

## Examples

```python
from torchflare.metrics import Accuracy

# Binary-Classification Problems
acc = Accuracy(num_classes=1, threshold=0.7, multilabel=False)

# Mutliclass-Classification Problems
multiclass_acc = Accuracy(num_classes=4, multilabel=False)

# Multilabel-Classification Problems
multilabel_acc = Accuracy(num_classes=5, multilabel=True, threshold=0.7)
```

::: torchflare.modules.am_softmax

``` python
import torch.nn as nn
from torchflare.modules import AMSoftmax

layer = AMSoftmax(in_features=1024, out_features=256, m=0.45, s=64)
crit = nn.CrossEntropyLoss()
logits = layer(emebedding, targets)
loss = crit(logits, targets)
```

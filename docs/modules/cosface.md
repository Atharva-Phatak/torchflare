::: torchflare.modules.cosface
    rendering:
         show_root_toc_entry: false

``` python
import torch.nn as nn
from torchflare.modules import CosFace

layer = CosFace(in_features=1024, out_features=256, m=0.45, s=64)
crit = nn.CrossEntropyLoss()
logits = layer(emebedding, targets)
loss = crit(logits, targets)
```

::: torchflare.datasets.text_dataset.TextClassificationDataset
    handler: python
    selection:
      members:
        - from_df
    rendering:
         show_root_toc_entry: false


## Examples

* ### from_df
``` python
import transformers
from torchflare.datasets import TextClassificationDataset

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

ds = TextClassificationDataset.from_df(
    df=df, input_col="tweet", label_cols="label", tokenizer=tokenizer, max_len=128
)
```

::: torchflare.datasets.tabular.TabularDataset
    handler: python
    selection:
      members:
        - from_df
        - from_csv


## Examples

* ### from_df
``` python
from torchflare.datasets import TabularDataset

ds = TabularDataset.from_df(df=df, feature_cols=["col1", "col2"], label_cols="labels")
```
***
* ### from_csv
``` python
from torchflare.datasets import TabularDataset

ds = TabularDataset.from_csv(
    csv_path="/train/train_data.csv", feature_cols=["col1", "col2"], label_cols="labels"
)
```

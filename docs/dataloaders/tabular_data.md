::: torchflare.datasets.tabular_dataloader.TabularDataloader
    handler: python
    selection:
       members:
         - from_df
         - from_csv
         - get_loader
    rendering:
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false


* ### from_df

``` python

from torchflare.datasets import TabularDataloader

dl = TabularDataloader.from_df(df=df,
                            feature_cols= ["col1" , "col2"],
                            label_cols="labels"
                            ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ### from_csv

``` python

from torchflare.datasets import TabularDataloader

dl = TabularDataloader.from_csv(csv_path="/train/train_data.csv",
                                feature_cols=["col1" , "col2"],
                                label_cols="labels"
                                ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```

::: torchflare.datasets.image_dataloader.ImageDataloader
    handler: python
    selection:
       members:
         - from_df
         - from_csv
         - from_folders
         - get_loader
    rendering:
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false

## Examples

* ###from_df

``` python

from torchflare.datasets import ImageDataloader

dl = ImageDataloader.from_df(df = train_df,
                              path = "/train/images",
                              image_col = "image_id",
                              label_cols="label",
                              augmentations=augs,
                              extension='.jpg'
                              ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ###from_folders

``` python

from torchflare.datasets import ImageDataloader

dl = ImageDataloader.from_folders(path="/train/images",
                                   augmentations=augs,
                                   convert_mode="RGB"
                              ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ###from_csv

``` python

from torchflare.datasets import ImageDataloader

dl = ImageDataloader.from_csv(csv_path = "./train/train.csv",
                              path = "/train/images",
                              image_col = "image_id",
                              label_cols="label",
                              augmentations=augs,
                              extension='.jpg'
                              ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

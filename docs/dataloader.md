::: torchflare.datasets.dataloaders.SimpleDataloader
    handler: python
    selection:
       members:
         - image_data_from_df
         - image_data_from_folders
         - segmentation_data_from_df
         - segmentation_data_from_folders
         - tabular_data_from_df
         - tabular_data_from_csv
         - text_classification_data_from_df
         - get_loaders

## Examples

* ###image_data_from_df

``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.image_data_from_df(df = train_df,
                              path = "/train/images",
                              image_col = "image_id",
                              label_cols="label",
                              augmentations=augs,
                              extension='./jpg'
                              ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ###image_data_from_folders

``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.image_data_from_folders(path="/train/images",
                                   augmentations=augs,
                                   convert_mode="RGB"
                              ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ###segmentation_data_from_rle
``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.segmentation_data_from_rle(df=df,
                path="/train/images",
                image_col="image_id",
                mask_cols=["EncodedPixles"],
                extension=".jpg",
                mask_size=(320,320),
                num_classes=4,
                augmentations=augs,
                image_convert_mode="RGB"
                ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```

***
* ###segmentation_data_from_folders
``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.segmentation_data_from_folders(
                    image_path="/train/images",
                    mask_path="/train/masks",
                    augmentations=augs,
                    image_convert_mode="L",
                    mask_convert_mode="L",
                    ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```

***

* ### tabular_data_from_df

``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.tabular_data_from_df(
                                df=df,
                                feature_cols= ["col1" , "col2"],
                                label_cols="labels"
                                ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```
***

* ### tabular_data_from_csv

``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.tabular_data_from_csv(
                                csv_path="/train/train_data.csv",
                                feature_cols=["col1" , "col2"],
                                label_cols="labels"
                                ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)
```

* ### text_classification_data_from_df

``` python

from torchflare.datasets import SimpleDataloader

dl = SimpleDataloader.text_classification_data_from_df(df=df,
                                           input_col="tweet",
                                            label_cols="label",
                                           tokenizer=tokenizer,
                                           max_len=128
                                           ).get_loader(batch_size=64, # Required Args.
                                                       shuffle=True, # Required Args.
                                                       num_workers = 0, # keyword Args.
                                                       collate_fn = collate_fn # keyword Args.
                                                       )

```

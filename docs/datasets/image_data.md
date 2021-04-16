::: torchflare.datasets.classification.ImageDataset
    handler: python
    selection:
      members:
        - from_df
        - from_folders



## Examples

* ### from_df
``` python
    from torchflare.datasets import ImageDataset

    ds = ImageDataset.from_df(df = train_df,
                              path = "/train/images",
                              image_col = "image_id",
                              label_cols="label",
                              augmentations=augmentations,
                              extension='./jpg'
                              convert_mode = "RGB"
                              )
```
***
* ### from_folders
``` python
    from torchflare.datasets import ImageDataset

    ds = ImageDataset.from_folders(path="/train/images",
                                   augmentations=augs,
                                   convert_mode="RGB"
                              )
```

::: torchflare.datasets.cross_val.CVSplit
    handler: python
    selection:
      members:
        - __init__
        - get_loaders
    rendering:
         show_root_toc_entry: false


## Examples

``` python
    from torchflare.datasets import CVSplit, ImageDataset

    n_splits = 5
    # Creating a PyTorch Dataset.
    ds = ImageDataset.from_df(df = train_df,
                              path = "/train/images",
                              image_col = "image_id",
                              label_cols="label",
                              augmentations=augmentations,
                              extension='.jpg'
                              convert_mode = "RGB"
                              )

    cv_data = CVSplit(dataset = ds, cv = "KFold",n_splits = n_splits,shuffle = True,
                      random_state = 42)

    for fold in range(n_splits):

        # Generate Train and validation dataloaders per fold.
        train_dl , valid_dl = cv_data.get_loaders(fold = fold,
                                        train_params=dict(batch_size = 64, shuffle = True,
                                                                    num_workers = 0),
                                        val_params=dict(batch_size = 32,shuffle = False))

```
***

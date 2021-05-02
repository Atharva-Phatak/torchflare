::: torchflare.datasets.segmentation_dataloader.SegmentationDataloader
    handler: python
    selection:
       members:
         - from_rle
         - from_folders
         - get_loader
    rendering:
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false



* ###from_rle
``` python

from torchflare.datasets import SegmentationDataloader

dl = SegmentationDataloader.from_rle(df=df,
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
* ###from_folders
``` python

from torchflare.datasets import SegmentationDataloader

dl = SegmentationDataloader.from_folders(
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

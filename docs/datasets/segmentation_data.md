::: torchflare.datasets.segmentation.SegmentationDataset
    handler: python
    selection:
      members:
        - from_rle
        - from_folders
    rendering:
         show_root_toc_entry: false


## Examples

* ### from_df
``` python
from torchflare.datasets import SegmentationDataset

ds = SegmentationDataset.from_rle(
    df=df,
    path="/train/images",
    image_col="image_id",
    mask_cols=["EncodedPixles"],
    extension=".jpg",
    mask_size=(320, 320),
    num_classes=4,
    augmentations=augs,
    image_convert_mode="RGB",
)
```
***
* ### from_folders
``` python
from torchflare.datasets import SegmentationDataset

ds = SegmentationDataset.from_folders(
    image_path="/train/images",
    mask_path="/train/masks",
    augmentations=augs,
    image_convert_mode="L",
    mask_convert_mode="L",
)
```

import glob
import pathlib
from typing import List, Optional, Tuple, Union

import albumentations as A
import pandas as pd

from torchflare.datasets.core_utils import (
    apply_image_augmentations,
    apply_segmentation_augs,
    get_iloc_cols,
    join_paths,
    make_masks,
    open_image,
)
from torchflare.datasets.data_core import BaseDataset, ItemReader


def get_rle(df: pd.DataFrame, image_col: str, mask_cols: List[str], name: str) -> List[str]:
    """Get rule length encoding for corresponding image ids."""
    ids = df[df[image_col] == name].index.values
    rle = df.loc[ids, mask_cols].values.tolist()
    return rle


def create_rle_list(
    df: pd.DataFrame,
    image_col: str,
    mask_cols: List[str],
) -> List[List[str]]:
    """Create mask list.

    Args:
        df : The dataframe.
        image_col: The column containing image_names.
        mask_cols : The column/columns containing mask rle's.

    Returns:
        A list of rle's
    """
    if isinstance(image_col, list) and len(image_col) == 1:
        image_col = image_col[0]

    image_names = df[image_col].values.tolist()
    mask_list = []

    for name in image_names:
        rle = get_rle(df, image_col, mask_cols, name)
        mask_list.append(rle)

    return mask_list


class MaskDataset(BaseDataset):
    """Dataset for image segmentation."""

    def __init__(
        self,
        image_convert_mode: str,
        mask_convert_mode: str,
        shape: Tuple = None,
        num_classes: int = None,
        **kwargs
    ):
        super(MaskDataset, self).__init__(**kwargs)
        self.mask_convert_mode = mask_convert_mode
        self.image_convert_mode = image_convert_mode
        self.shape = shape
        self.num_classes = num_classes

    def _get_labels(self, idx):
        if any(isinstance(ele, list) for ele in self.y):
            mask = make_masks(rle=self.y[idx], shape=self.shape, num_classes=self.num_classes)
        else:
            mask = open_image(self.y[idx], convert_mode=self.mask_convert_mode)
        return mask

    def __getitem__(self, idx):
        x = open_image(self.item_reader.items[idx], convert_mode=self.image_convert_mode)
        if not self.is_y_none:
            mask = self._get_labels(idx=idx)
            images, mask = apply_segmentation_augs(
                image=x, transforms=self.item_reader.transforms, mask=mask
            )
            return images, mask
        images = apply_image_augmentations(x, transforms=self.item_reader.transforms)
        return images


class SegmentationDataset(ItemReader):
    """PyTorch style dataset for image segmentation."""

    def __init__(self, input_cols, image_convert_mode, **kwargs):
        super(SegmentationDataset, self).__init__(**kwargs)
        self.image_convert_mode = image_convert_mode
        self.input_cols = input_cols
        self.mask_dataset = MaskDataset

    def apply_target_transforms(self, transforms, item):
        """Method to apply transforms to inputs."""
        raise NotImplementedError

    def apply_input_transforms(self, transforms, item):
        """Method to apply transforms to targets."""
        raise NotImplementedError

    # skipcq : PYL-W0221
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        path: Union[str, pathlib.Path],
        input_columns: List[str],
        transforms: A.Compose = None,
        image_convert_mode: str = "RGB",
        extension: Optional[str] = None,
        **kwargs
    ):
        """Method to read images from dataframe.

        Args:
            df : The dataframe containing the image names/ids.
            input_columns : A list containing columns which
                        have names of images.
            path: The path where images are saved.
            transforms: The transforms to be used on the inputs.
            image_convert_mode: The mode to be passed to PIL.Image.convert.
            extension : The extension of image file.

        Example:

            .. code-block:: python

                from torchflare.datasets import SegmentationDataset
                ds = SegmentationDataset.from_df(
                    df=df,
                    path="/train/images",
                    input_columns=["image_id"],
                    extension=".jpg",
                    augmentations=augs,
                    image_convert_mode="RGB",
                ).masks_from_rle(mask_cols=["EncodedPixles"],
                                mask_size=(320, 320),
                                num_classes=4)

        """
        path = pathlib.Path(path)
        files = get_iloc_cols(df, input_columns)
        files = join_paths(path=path, files=files, extension=extension)
        return cls(
            items=files,
            transforms=transforms,
            df=df,
            path=path,
            image_convert_mode=image_convert_mode,
            input_cols=input_columns,
            **kwargs
        )

    # skipcq : PYL-W0221
    @classmethod
    def from_folders(
        cls,
        image_path: Union[str, pathlib.Path],
        transforms: Optional[A.Compose] = None,
        image_convert_mode: str = "RGB",
        extension: str = None,
        **kwargs
    ):
        """Classmethod to create pytorch dataset from folders.

        Args:
            image_path: The path where images are stored.
            transforms: The transforms to apply on images and masks.
            image_convert_mode: The mode to be passed to PIL.Image.convert for input images
            extension : The extension for image like .jpg, etc

        Example:

            .. code-block:: python

                from torchflare.datasets import SegmentationDataset
                ds = SegmentationDataset.from_folders(
                        image_path="/train/images",
                        transforms=augs,
                        image_convert_mode="L",
                    ).masks_from_folders(mask_convert_mode="L",
                    mask_path="/train/masks",
                    mask_convert_mode = "L")
        """
        files = glob.glob(image_path + "/*")
        return cls(
            items=files,
            path=image_path,
            transforms=transforms,
            input_cols=None,
            image_convert_mode=image_convert_mode,
            extension=extension,
            **kwargs
        )

    def _create_mask_dataset(
        self, labels, shape=None, num_classes=None, mask_convert_mode=None, target_transforms=None
    ):
        return self.mask_dataset(
            item_reader=self,
            y=labels,
            target_transforms=target_transforms,
            mask_convert_mode=mask_convert_mode,
            image_convert_mode=self.image_convert_mode,
            shape=shape,
            num_classes=num_classes,
        )

    def masks_from_rle(
        self, shape: Tuple[int, int], num_classes: int, mask_columns: Optional[List[str]]
    ):
        """Create masks from rule length encoding.

        Args:
            mask_columns : The list of columns containing the rule length encoding.
            shape : The shape for masks.
            num_classes: The number of num_classes
        """
        masks = create_rle_list(
            df=self.df,
            image_col=self.input_cols,
            mask_cols=mask_columns,
        )
        return self._create_mask_dataset(
            labels=masks,
            shape=shape,
            num_classes=num_classes,
            mask_convert_mode=None,
            target_transforms=None,
        )

    def masks_from_folders(self, mask_path: Union[str, pathlib.Path], mask_convert_mode: str):
        """Read masks from folders.

        Args:
            mask_path: The path where masks are stored.
            mask_convert_mode: The mode to be passed to PIL.Image.convert for masks.
        """
        masks = glob.glob(mask_path + "/*")
        return self._create_mask_dataset(
            labels=masks, mask_convert_mode=mask_convert_mode, target_transforms=None
        )

    def add_test(self):
        """Method to create dataset for inference."""
        return self._create_mask_dataset(
            labels=None, mask_convert_mode=None, target_transforms=None
        )


__all__ = ["SegmentationDataset"]

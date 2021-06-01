"""Implements dataset for Image segmentation tasks."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import albumentations as A
import pandas as pd
import torchvision
from torch.utils.data import Dataset

from torchflare.datasets.utils import apply_image_transforms, apply_segmentation_augs, make_masks, open_image


def get_rle(df: pd.DataFrame, image_col: str, mask_cols: List[str], name: str) -> List[str]:
    ids = df[df[image_col] == name].index.values
    rle = df.loc[ids, mask_cols].values.tolist()
    return rle


class SegmentationDataset(Dataset):
    """Class to create a dataset for image segmentation tasks."""

    def __init__(
        self,
        image_paths_list: List,
        mask_list=None,
        augmentations: [Union[A.Compose, torchvision.transforms.Compose]] = None,
        image_convert_mode: str = "RGB",
        mask_convert_mode: str = "L",
        **kwargs
    ):
        """Constructor method for segmentation Dataset.

        Args:
            image_paths_list:  A list containing paths to all images. Ex ['/train/img.png' , 'train/img_0.png' ,etc]
            mask_list: A list containing all the masks.
            augmentations: Augmentations to be applied to image , masks.
            image_convert_mode:  The mode in which images should be opened.
            mask_convert_mode: The mode in which masks should be opened.
            **kwargs: Extra named args.
        """
        self.inputs = image_paths_list
        self.labels = mask_list
        self.augmentations = augmentations
        self.kwargs = kwargs
        self.image_convert_mode = image_convert_mode
        self.mask_convert_mode = mask_convert_mode

    def __len__(self):
        """__len__ method.

        Returns:
            The length of dataloader.
        """
        return len(self.inputs)

    def _get_labels(self, idx):
        if any(isinstance(ele, list) for ele in self.labels):
            mask = make_masks(self.labels[idx], **self.kwargs)
        else:
            mask = open_image(self.labels[idx], convert_mode=self.mask_convert_mode)

        return mask

    def __getitem__(self, item):
        """__getitem__ method.

        Args:
            item : The selected id.

        Returns:
            Tensors of images , masks if labels is provided else Tensors of images.
        """
        images = open_image(self.inputs[item], convert_mode=self.image_convert_mode)
        if self.labels is not None:
            mask = self._get_labels(idx=item)
            images, mask = apply_segmentation_augs(images, augs=self.augmentations, mask=mask)
            return images, mask

        else:
            images = apply_image_transforms(images, augs=self.augmentations)
            return images

    @staticmethod
    def _join_paths(path: str, file_names: List[str], extension: Optional[str] = None) -> List[str]:

        if extension is None:
            return [os.path.join(path, x) for x in file_names]

        return [os.path.join(path, x + extension) for x in file_names]

    @staticmethod
    def create_mask_list(df: pd.DataFrame, image_col: str, mask_cols: List[str]) -> List[List[str]]:
        """Create mask list.

        Args:
            df : The dataframe.
            image_col: The column containing image_names.
            mask_cols : The column/columns containing mask rle's.

        Returns:
            A list of masks
        """
        image_names = df[image_col].values.tolist()
        mask_list = []

        for name in image_names:
            mask_list.append(get_rle(df, image_col, mask_cols, name))

        return mask_list

    @classmethod
    def from_rle(
        cls,
        path: str,
        df: pd.DataFrame,
        image_col: str,
        mask_cols: List[str] = None,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        mask_size: Tuple[int, int] = None,
        num_classes: int = None,
        extension: str = None,
        image_convert_mode: str = "RGB",
    ):
        """Classmethod to create pytorch dataset when you have rule length encodings for masks stored in a dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            image_col: The name of the image column containing the image name/ids.
            augmentations: The batch_mixers to be used on images and the masks.
            mask_cols: The list of columns containing the rule length encoding.
            mask_size: The size of mask.
            num_classes: The number of num_classes.
            image_convert_mode: The mode to be passed to PIL.Image.convert.
            extension : The extension of image file.
                If your image_names do not have extension then set extension to '.jpg' or '.png' ,etc

        Returns:
            returns image_paths_list , labels , image_convert_mode , augmentations and extra kwargs.

        Note:
            If you want to create a dataset for testing set mask_cols = None, mask_size = None, num_classes = None.
            The created masks will be binary.

        Examples:
            .. code-block:: python

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

        """
        image_list = cls._join_paths(path=path, file_names=df[image_col].values.tolist(), extension=extension)

        mask_list = (
            cls.create_mask_list(df=df, image_col=image_col, mask_cols=mask_cols) if mask_cols is not None else None
        )
        return cls(
            image_list,
            mask_list,
            augmentations,
            image_convert_mode=image_convert_mode,
            shape=mask_size,
            num_classes=num_classes,
        )

    @classmethod
    def from_folders(
        cls,
        image_path: str,
        mask_path: str = None,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        image_convert_mode: str = "L",
        mask_convert_mode: str = "L",
    ):
        """Classmethod to create pytorch dataset from folders.

        Args:
            image_path: The path where images are stored.
            mask_path: The path where masks are stored.
            augmentations: The batch_mixers to apply on images and masks.
            image_convert_mode: The mode to be passed to PIL.Image.convert for input images
            mask_convert_mode: The mode to be passed to PIL.Image.convert for masks.

        Returns:
            returns image_paths_list , mask_path , augmentations , image_convert_mode , mask_convert_mode.

        Note:
             If you want to create a dataset for testing just set mask_path = None.


        Examples:
            .. code-block:: python

                from torchflare.datasets import SegmentationDataset
                ds = SegmentationDataset.from_folders(
                    image_path="/train/images",
                    mask_path="/train/masks",
                    augmentations=augs,
                    image_convert_mode="L",
                    mask_convert_mode="L",
                )

        """
        image_files = cls._join_paths(image_path, os.listdir(image_path))
        mask_files = cls._join_paths(mask_path, os.listdir(mask_path)) if mask_path is not None else None

        return cls(
            image_paths_list=image_files,
            mask_list=mask_files,
            augmentations=augmentations,
            image_convert_mode=image_convert_mode,
            mask_convert_mode=mask_convert_mode,
        )


__all__ = ["SegmentationDataset"]

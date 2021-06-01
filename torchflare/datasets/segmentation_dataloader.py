"""Wrapper for dataloaders."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import albumentations as A
import pandas as pd
import torchvision
from torch.utils.data import DataLoader

from torchflare.datasets.segmentation import SegmentationDataset


class SegmentationDataloader:
    """Class to create easy to use dataloaders."""

    def __init__(self, ds):
        """Constructor method.

        Args:
            ds : A pytorch style dataset having __len__ and __getitem__ methods.
        """
        self.ds = ds

    @classmethod
    def from_rle(
        cls,
        path: str,
        df: pd.DataFrame,
        image_col: str,
        mask_cols: List[str] = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        mask_size: Tuple[int, int] = None,
        num_classes: List = None,
        extension: Optional[str] = None,
        image_convert_mode: str = "RGB",
    ):
        """Classmethod to create a dataset for segmentation when you have, rule length encodings stored in a dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            image_col: The name of the image column containing the image name/ids along with image extension.
                i.e. the images should have names like img_215.jpg or img_name.png ,etc
            augmentations: The batch_mixers to be used on images and the masks.
            mask_cols: The list of columns containing the rule length encoding.
            mask_size: The size of mask.
            num_classes: The list of num_classes.
            extension : The image file extension.
            image_convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            Pytorch dataset created from Rule-length encodings

        Note:

            This method will make only binary masks.

            If you want to create a dataloader for testing set mask_cols = None, mask_size = None, num_classes = None.

        Examples:

            .. code-block:: python

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


        """
        return cls(
            SegmentationDataset.from_rle(
                path=path,
                df=df,
                image_col=image_col,
                mask_cols=mask_cols,
                augmentations=augmentations,
                mask_size=mask_size,
                num_classes=num_classes,
                image_convert_mode=image_convert_mode,
                extension=extension,
            )
        )

    @classmethod
    def from_folders(
        cls,
        image_path: str,
        mask_path: str = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
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
            Pytorch Segmentation dataset created from folders.

        Example:

            .. code-block:: python

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

        """
        return cls(
            SegmentationDataset.from_folders(
                image_path=image_path,
                mask_path=mask_path,
                augmentations=augmentations,
                image_convert_mode=image_convert_mode,
                mask_convert_mode=mask_convert_mode,
            )
        )

    def get_loader(self, batch_size: int = 32, shuffle: bool = True, **dl_params) -> DataLoader:
        """Method to get dataloader.

        Args:
            batch_size: The batch size to use
            shuffle: Whether to shuffle the inputs.
            **dl_params : Keyword arguments related to dataloader

        Returns:
            A PyTorch dataloader with given arguments.
        """
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle, **dl_params)
        return dl


__all__ = ["SegmentationDataloader"]

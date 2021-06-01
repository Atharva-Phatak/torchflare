"""Wrapper for dataloaders."""
from __future__ import annotations

from typing import List, Optional, Union

import albumentations as A
import pandas as pd
import torchvision
from torch.utils.data import DataLoader

from torchflare.datasets.image_classification import ImageDataset


class ImageDataloader:
    """Class to create easy to use dataloaders."""

    def __init__(self, ds):
        """Constructor method.

        Args:
            ds : A pytorch style dataset having __len__ and __getitem__ methods.
        """
        self.ds = ds

    @classmethod
    def from_df(
        cls,
        path: str,
        df: pd.DataFrame,
        image_col: str,
        label_cols: Union[str, List[str]] = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
        extension: str = None,
    ):
        """Classmethod to create a dataset for image data when you have image names/ids , labels in dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            image_col: The name of the image column containing the image name/ids along with image extension.
                i.e. the images should have names like img_215.jpg or img_name.png ,etc
            augmentations: The batch_mixers to be used on images.
            label_cols: The list of columns containing targets.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            Pytorch dataset created from dataframe.

        Note:

            For inference do not pass in the label_cols, keep it None.

            Augmentations must be Compose objects from albumentations or torchvision.

        Examples:

            .. code-block:: python

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
        """
        return cls(
            ImageDataset.from_df(
                path=path,
                df=df,
                image_col=image_col,
                label_cols=label_cols,
                augmentations=augmentations,
                convert_mode=convert_mode,
                extension=extension,
            )
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        csv_path: str,
        image_col: str,
        label_cols: Union[str, List[str]] = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
        extension: str = None,
    ):
        """Classmethod to create a dataset for image data when you have image names/ids , labels in a csv.

        Args:
            path: The path where images are saved.
            csv_path: The full path to csv. Example: ./train/train_data.csv
            image_col: The name of the image column containing the image name/ids along with image extension
                i.e. the images should have names like img_215.jpg or img_name.png ,etc
            augmentations: The batch_mixers to be used on images.
            label_cols: The list of columns containing targets.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            Pytorch dataset created from dataframe.

        Note:

            For inference do not pass in the label_cols, keep it None.
            Augmentations must be Compose objects from albumentations or torchvision.

        Examples:

            .. code-block:: python

                from torchflare.datasets import ImageDataloader
                dl = ImageDataloader.from_csv(csv_path = "train/train.csv",
                                  path = "/train/images",
                                  image_col = "image_id",
                                  label_cols="label",
                                  augmentations=augs,
                                  extension='.jpg'
                                  ).get_loader(batch_size=64, # Required Args.
                                               shuffle=True, # Required Args.
                                               num_workers = 0, # keyword Args.
                                               collate_fn = collate_fn # keyword Args.)
        """
        return cls(
            ImageDataset.from_df(
                path=path,
                df=pd.read_csv(csv_path),
                image_col=image_col,
                label_cols=label_cols,
                augmentations=augmentations,
                convert_mode=convert_mode,
                extension=extension,
            )
        )

    @classmethod
    def from_folders(
        cls,
        path: str,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
    ):
        """Classmethod to create pytorch dataset from folders.

        Args:
            path: The path where images are stored.
            augmentations:The batch_mixers to be used on images.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            Pytorch image dataset created from folders

        Note:

            Augmentations must be Compose objects from albumentations or torchvision.

            The training directory structure should be as follows:
                train/class_1/xxx.jpg
                .
                .
                train/class_n/xxz.jpg

            The test directory structure should be as follows:
                test_dir/xxx.jpg
                test_dir/xyz.jpg
                test_dir/ppp.jpg

        Examples:
            .. code-block:: python

                from torchflare.datasets import ImageDataloader
                dl = ImageDataloader.from_folders(path="/train/images",
                                       augmentations=augs,
                                       convert_mode="RGB"
                                  ).get_loader(batch_size=64, # Required Args.
                                               shuffle=True, # Required Args.
                                               num_workers = 0, # keyword Args.
                                               collate_fn = collate_fn # keyword Args.)
        """
        return cls(ImageDataset.from_folders(path=path, augmentations=augmentations, convert_mode=convert_mode))

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


__all__ = ["ImageDataloader"]

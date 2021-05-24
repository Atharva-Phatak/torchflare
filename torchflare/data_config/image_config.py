"""Implements Image Data configs."""
from dataclasses import MISSING, dataclass, field
from typing import List, Union

import albumentations as A
import pandas as pd
import torchvision

from torchflare.data_config.base import BaseConfig
from torchflare.datasets.image_classification import ImageDataset


@dataclass
class _ImageDataConfigDF:

    path: str = field(default=MISSING, metadata={"help": "The path where images are saved."})
    df: pd.DataFrame = field(
        default=MISSING, metadata={"help": "The dataframe containing the image name/ids, and the targets."}
    )
    image_col: str = field(
        default=MISSING, metadata={"help": "The name of the image column containing the image name/ids."}
    )

    label_cols: Union[str, List[str]] = field(
        default=None, metadata={"help": "Column name or list of column names containing targets."}
    )
    augmentations: Union[A.Compose, torchvision.transforms.Compose] = field(
        default=None, metadata={"help": "The augmentations to be used on images."}
    )
    extension: str = field(default=None, metadata={"help": "The image file extension."})
    convert_mode: str = field(default="RGB", metadata={"help": "The mode to be passed to PIL.Image.convert."})


@dataclass
class _ImageDataConfigFolders:
    path: str = field(default=MISSING, metadata={"help": "The path where images are saved."})
    augmentations: Union[A.Compose, torchvision.transforms.Compose] = field(
        default=None, metadata={"help": "The augmentations to be used on images."}
    )
    convert_mode: str = field(default="RGB", metadata={"help": "The mode to be passed to PIL.Image.convert."})


class ImageDataConfig(BaseConfig):
    """Class to create data configs for image data."""

    def __init__(self, config, data_method):
        """Constructor Method.

        Args:
            config: The config object.
            data_method: The method which will be used to create the dataset.
        """
        super(ImageDataConfig, self).__init__(config=config, data_method=data_method)

    @classmethod
    def from_df(
        cls,
        path: str,
        df: pd.DataFrame,
        image_col: str,
        label_cols: Union[str, List[str]] = None,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        extension: str = None,
        convert_mode: str = "RGB",
    ):
        """Classmethod to create data config if you want to load data from dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            image_col: The name of the image column containing the image name/ids.
            augmentations: The augmentations to be used on images.
            label_cols: Column name or list of column names containing targets.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
             returns config object and a method to create a Pytorch-style dataset.

        Note:
            For inference do not pass in the label_cols, keep it None.

            Augmentations : They must be Compose objects from albumentations or torchvision.
                When using albumentations do not use ToTensorV2().

            extension : If you specify extension be it jpg,png,etc. Please include '.' in extension
                i.e. '.jpg' or '.png'.
        """
        return cls(
            config=_ImageDataConfigDF(
                path=path,
                df=df,
                image_col=image_col,
                label_cols=label_cols,
                augmentations=augmentations,
                extension=extension,
                convert_mode=convert_mode,
            ),
            data_method=ImageDataset.from_df,
        )

    @classmethod
    def from_folders(
        cls,
        path: str,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        convert_mode: str = "RGB",
    ):
        """Classmethod to create data config if you want to load data from folders.

        Args:
            path: The path where images are stored.
            augmentations:The batch_mixers to be used on images.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            return image_paths_list , labels_list , augmentations and convert_mode.

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
        """
        return cls(
            _ImageDataConfigFolders(path=path, augmentations=augmentations, convert_mode=convert_mode),
            data_method=ImageDataset.from_folders,
        )


__all__ = ["ImageDataConfig"]

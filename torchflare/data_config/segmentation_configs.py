"""Implements Segmentation Data configs."""
from dataclasses import MISSING, dataclass, field
from typing import List, Tuple, Union

import albumentations as A
import pandas as pd
import torchvision

from torchflare.data_config.base import BaseConfig
from torchflare.datasets.segmentation import SegmentationDataset


@dataclass
class _SegmentationDataConfigRLE:
    path: str = field(default=MISSING, metadata={"help": "The path where images are saved."})
    df: pd.DataFrame = field(
        default=MISSING, metadata={"help": "The dataframe containing the image name/ids, and the targets."}
    )
    image_col: str = field(
        default=MISSING, metadata={"help": "The name of the image column containing the image name/ids."}
    )
    mask_cols: List[str] = field(
        default=None, metadata={"help": "The list of columns containing the rule length encoding."}
    )
    augmentations: Union[A.Compose, torchvision.transforms.Compose] = field(
        default=None, metadata={"help": "The augmentations to be used on images."}
    )
    mask_size: Tuple[int, int] = field(default=None, metadata={"help": "The size of mask."})
    num_classes: int = field(default=None, metadata={"help": "The number of num_classes."})
    extension: str = field(default=None, metadata={"help": "The image file extension."})
    image_convert_mode: str = field(default="RGB", metadata={"help": "The mode to be passed to PIL.Image.convert."})


@dataclass
class _SegmentationDataConfigFolders:
    image_path: str = field(default=MISSING, metadata={"help": "The path where images are saved."})
    mask_path: str = field(default=MISSING, metadata={"help": "The path where masks are saved."})
    augmentations: Union[A.Compose, torchvision.transforms.Compose] = field(
        default=None, metadata={"help": "The augmentations to be used on images."}
    )
    image_convert_mode: str = field(
        default="L", metadata={"help": "The mode to be passed to PIL.Image.convert for input images."}
    )
    mask_convert_mode: str = field(
        default="L", metadata={"help": "The mode to be passed to PIL.Image.convert for input masks."}
    )


class SegmentationDataConfig(BaseConfig):
    """Class to create configs for Segmentation data."""

    def __init__(self, config, data_method):
        """Constructor Method.

        Args:
            config: The config object.
            data_method: The method which will be used to create the dataset.
        """
        super(SegmentationDataConfig, self).__init__(config=config, data_method=data_method)

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
    ) -> "SegmentationDataConfig":
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
               returns config object and a method to create a Pytorch-style dataset.

           Note:
               If you want to create a dataset for testing set mask_cols = None, mask_size = None, num_classes = None.
               The created masks will be binary.
        """
        return cls(
            _SegmentationDataConfigRLE(
                path=path,
                df=df,
                image_col=image_col,
                mask_cols=mask_cols,
                augmentations=augmentations,
                mask_size=mask_size,
                num_classes=num_classes,
                extension=extension,
                image_convert_mode=image_convert_mode,
            ),
            data_method=SegmentationDataset.from_rle,
        )

    @classmethod
    def from_folders(
        cls,
        image_path: str,
        mask_path: str = None,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        image_convert_mode: str = "L",
        mask_convert_mode: str = "L",
    ) -> "SegmentationDataConfig":
        """Classmethod to create pytorch dataset from folders.

        Args:
            image_path: The path where images are stored.
            mask_path: The path where masks are stored.
            augmentations: The batch_mixers to apply on images and masks.
            image_convert_mode: The mode to be passed to PIL.Image.convert for input images
            mask_convert_mode: The mode to be passed to PIL.Image.convert for masks.

        Returns:
                returns config object and a method to create a Pytorch-style dataset.

        Note:
            If you want to create a dataset for testing just set mask_path = None.
        """
        return cls(
            _SegmentationDataConfigFolders(
                mask_path=mask_path,
                augmentations=augmentations,
                image_convert_mode=image_convert_mode,
                mask_convert_mode=mask_convert_mode,
                image_path=image_path,
            ),
            data_method=SegmentationDataset.from_folders,
        )


__all__ = ["SegmentationDataConfig"]

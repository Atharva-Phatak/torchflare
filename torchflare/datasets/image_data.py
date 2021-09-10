import glob
import pathlib
from typing import Callable, List, Optional, Union

import albumentations as A
import pandas as pd
import torch

from torchflare.datasets.core_utils import (
    apply_image_augmentations,
    get_iloc_cols,
    join_paths,
    open_image,
    to_tensor,
)
from torchflare.datasets.data_core import ItemReader


class ImageDataset(ItemReader):
    """Class to create the dataset for Image Classification."""

    def __init__(self, convert_mode: str, *args, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)
        self.convert_mode = convert_mode

    def apply_input_transforms(self, transforms: A.Compose, item) -> torch.Tensor:
        """Method to apply augmentations to images."""
        image = open_image(x=item, convert_mode=self.convert_mode)
        if transforms is not None:
            image = apply_image_augmentations(image, transforms)
        return to_tensor(image)

    def apply_target_transforms(self, transforms: Union[A.Compose, Callable], item) -> torch.Tensor:
        """Method to apply transformations on targets."""
        if transforms is not None:
            item = transforms(item)
        return to_tensor(item)

    # skipcq : PYL-W0221
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        path: Union[str, pathlib.Path],
        input_columns: List[str],
        transforms: Optional[A.Compose] = None,
        convert_mode: str = "RGB",
        extension: str = None,
        **kwargs
    ):
        """Classmethod to read inputs from the given dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            input_columns: A list containing name/names of the
                            image columns containing the image name/ids.
            transforms: The augmentations to be used on images.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Example:

            .. code-block:: python

                from torchflare.datasets import ImageDataset

                ds = ImageDataset.from_df(df = df,
                    path = "train/images",
                    input_columns = ['image_ids'],
                    transforms = A.Compose([A.Resize(256,256)]
                ).targets_from_df(target_columns = ["targets"])
        """
        path = pathlib.Path(path)
        files = get_iloc_cols(df, input_columns)
        files = join_paths(path=path, files=files, extension=extension)
        return cls(
            items=files,
            transforms=transforms,
            df=df,
            path=path,
            convert_mode=convert_mode,
            **kwargs
        )

    # skipcq : PYL-W0221
    @classmethod
    def from_folders(
        cls,
        path: Union[str, pathlib.Path],
        transforms: Optional[A.Compose] = None,
        convert_mode: str = "RGB",
        **kwargs
    ):
        """Classmethod to create pytorch dataset from folders.

        Args:
                path: The path where images are stored.
                transforms: The transforms to be applied to images.
                convert_mode: The mode to be passed to PIL.Image.convert.

        Note:
            Augmentations must be Compose objects from albumentations.

            The training directory structure should be as follows:
                train/class_1/xxx.jpg
                .
                .
                train/class_n/xxz.jpg
            The test directory structure should be as follows:
                test_dir/xxx.jpg
                test_dir/xyz.jpg
                test_dir/ppp.jpg

        Example:

            .. code-block:: python

                from torchflare.datasets import ImageDataset

                import albumentations as A

                ds = ImageDataset.from_folders(
                    path="/train/images",
                    transforms=A.Compose[A.Resize(256, 256)],
                    convert_mode="RGB"
                ).targets_from_folders(target_path="/train/images")
        """
        files = glob.glob(path + "/*/*")
        return cls(
            items=files, path=path, transforms=transforms, convert_mode=convert_mode, **kwargs
        )

    # skipcq : PYL-W0221
    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, pathlib.Path],
        path: Union[str, pathlib.Path],
        input_columns: List[str],
        transforms: Optional[A.Compose] = None,
        convert_mode: str = "RGB",
        extension: Optional[str] = None,
        **kwargs
    ):
        """Classmethod to read inputs from the given csv.

        Args:
            path: The path where images are saved.
            csv_path : Full path to the csv file.
            input_columns: A list containing names of the
                               image columns containing the image name/ids.
            transforms: The augmentations to be used on images.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Example:

            .. code-block:: python

                from torchflare.datasets import ImageDataset
                import albumentations as A

                ds = ImageDataset.from_csv(csv_path = "train/train.csv",
                    path = "train/images",
                    input_columns = ['image_ids'],
                    transforms = A.Compose([A.Resize(256,256)]
                   ).targets_from_df(target_columns = ["targets"])
        """
        df = pd.read_csv(csv_path)
        return cls.from_df(
            df=df,
            transforms=transforms,
            input_columns=input_columns,
            convert_mode=convert_mode,
            extension=extension,
            path=path,
        )


__all__ = ["ImageDataset"]

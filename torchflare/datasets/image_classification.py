"""Implements Image datasets for classification."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import pandas as pd
import torchvision
from torch.utils.data import Dataset

from torchflare.datasets.utils import apply_image_transforms, open_image, to_tensor


def get_files(directory: str):
    directory = os.path.expanduser(directory)
    return [os.path.join(directory, fname) for fname in os.listdir(directory)]


def get_files_and_labels(directory: str, class_to_idx: Dict[str, int]) -> Tuple[List, List]:
    image_list = []
    labels = []

    directory = os.path.expanduser(directory)

    for target_class in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)

        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                image_list.append(path)
                labels.append(class_idx)

    return image_list, labels


class ImageDataset(Dataset):
    """Class to create the dataset for Image Classification."""

    def __init__(
        self,
        image_paths_list: List,
        label_list: Optional[List] = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
    ):
        """Constructor method for Image Dataset class.

        Args:
            image_paths_list: A list containing paths to all images. Ex ['/train/img.png' , 'train/img_0.png' ,etc]
            label_list: A list containing all the labels.
            augmentations : The augmentations to apply to the images.
            convert_mode: The mode in which the image is opened.
        """
        self.labels = label_list
        self.inputs = image_paths_list
        self.augmentations = augmentations
        self.convert_mode = convert_mode

    @staticmethod
    def _get_labels_from_folders(path: str):
        classes = [d.name for d in os.scandir(path) if d.is_dir()]
        classes.sort()
        return {class_name: i for i, class_name in enumerate(classes)}

    @staticmethod
    def _get_labels_from_df(df: pd.DataFrame, label_cols: Union[str, List[str]]):

        return df.loc[:, label_cols].values.tolist() if label_cols is not None else None

    @staticmethod
    def _join_paths(path: str, file_names: List[str], extension: Optional[str] = None) -> List[str]:

        if extension is None:
            return [os.path.join(path, x) for x in file_names]

        return [os.path.join(path, x + extension) for x in file_names]

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
        """Classmethod to create pytorch dataset from the given dataframe.

        Args:
            path: The path where images are saved.
            df: The dataframe containing the image name/ids, and the targets
            image_col: The name of the image column containing the image name/ids.
            augmentations: The augmentations to be used on images.
            label_cols: Column name or list of column names containing targets.
            extension : The image file extension.
            convert_mode: The mode to be passed to PIL.Image.convert.

        Returns:
            return image_paths_list , labels_list , augmentations and convert_mode.

        Note:
            For inference do not pass in the label_cols, keep it None.

            Augmentations : They must be Compose objects from albumentations or torchvision.
                When using albumentations do not use ToTensorV2().

            extension : If you specify extension be it jpg,png,etc. Please include '.' in extension
                i.e. '.jpg' or '.png'.

        Examples:
            .. code-block:: python

                from torchflare.datasets import ImageDataset
                ds = ImageDataset.from_df(df = train_df,
                                            path = "/train/images",
                                            image_col = "image_id",
                                            label_cols="label",
                                            augmentations=augmentations,
                                            extension='./jpg'
                                            convert_mode = "RGB"
                                            )

        """
        img_list = cls._join_paths(path=path, file_names=df.loc[:, image_col].values, extension=extension)
        label_list = cls._get_labels_from_df(df=df, label_cols=label_cols)
        return cls(
            image_paths_list=img_list,
            label_list=label_list,
            augmentations=augmentations,
            convert_mode=convert_mode,
        )

    @classmethod
    def from_folders(
        cls,
        path: str,
        augmentations: Union[A.Compose, torchvision.transforms.Compose] = None,
        convert_mode: str = "RGB",
    ):
        """Classmethod to create pytorch dataset from folders.

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

        Examples:

            .. code-block:: python

                from torchflare.datasets import ImageDataset

                ds = ImageDataset.from_folders(path="/train/images",
                                   augmentations=augs,
                                   convert_mode="RGB"
                              )


        """
        class_to_idx = cls._get_labels_from_folders(path)
        if class_to_idx:
            image_list, label_list = get_files_and_labels(path, class_to_idx)
        else:
            image_list, label_list = get_files(path), None

        return cls(
            image_paths_list=image_list,
            label_list=label_list,
            augmentations=augmentations,
            convert_mode=convert_mode,
        )

    def __len__(self):
        """__len__ method.

        Returns:
            length of dataloader.
        """
        return len(self.inputs)

    def __getitem__(self, item: int):
        """__getitem__ method.

        Args:
            item: The id

        Returns:
            Tensors of Image , labels if labels are present else Tensors of Images.
        """
        images = open_image(self.inputs[item], convert_mode=self.convert_mode)
        images = apply_image_transforms(images, augs=self.augmentations)
        if self.labels is None:
            return images

        labels = self.labels[item]
        return images, to_tensor(labels)


__all__ = ["ImageDataset"]

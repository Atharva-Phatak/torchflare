"""Wrapper for dataloaders."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import albumentations as A
import pandas as pd
import torchvision
from torch.utils.data import DataLoader

from torchflare.datasets.classification import ImageDataset
from torchflare.datasets.segmentation import SegmentationDataset
from torchflare.datasets.tabular import TabularDataset
from torchflare.datasets.text_dataset import TextClassificationDataset


class SimpleDataloader:
    """Class to create easy to use dataloaders."""

    def __init__(self, ds):
        """Constructor method.

        Args:
            ds : A pytorch style dataset having __len__ and __getitem__ methods.
        """
        self.ds = ds

    @classmethod
    def image_data_from_df(
        cls,
        path: str,
        df: pd.DataFrame,
        image_col: str,
        label_cols: List[str] = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
        extension: str = None,
    ) -> SimpleDataloader:
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
    def image_data_from_folders(
        cls,
        path: str,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        convert_mode: str = "RGB",
    ) -> SimpleDataloader:
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

        """
        return cls(ImageDataset.from_folders(path=path, augmentations=augmentations, convert_mode=convert_mode))

    @classmethod
    def segmentation_data_from_rle(
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
    ) -> SimpleDataloader:
        """Classmethod to create a dataset for segmentation data when you have, rule length encodings stored in a dataframe.

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
    def segmentation_data_from_folders(
        cls,
        image_path: str,
        mask_path: str = None,
        augmentations: Optional[Union[A.Compose, torchvision.transforms.Compose]] = None,
        image_convert_mode: str = "L",
        mask_convert_mode: str = "L",
    ) -> SimpleDataloader:
        """Classmethod to create pytorch dataset from folders.

        Args:
            image_path: The path where images are stored.
            mask_path: The path where masks are stored.
            augmentations: The batch_mixers to apply on images and masks.
            image_convert_mode: The mode to be passed to PIL.Image.convert for input images
            mask_convert_mode: The mode to be passed to PIL.Image.convert for masks.

        Returns:
            Pytorch Segmentation dataset created from folders.
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

    @classmethod
    def tabular_data_from_df(
        cls, df: pd.DataFrame, feature_cols: Union[str, List[str]], label_cols: Optional[Union[str, List[str]]] = None,
    ) -> SimpleDataloader:
        """Classmethod to create dataset for tabular data from dataframe.

        Args:
            df: The dataframe containing features and labels.
            feature_cols: name(str) or list containing names feature columns.
            label_cols: name(str) or list containing names label columns.

        Returns:
            Tabular pytorch dataset
        """
        return cls(TabularDataset.from_df(df=df, feature_cols=feature_cols, label_cols=label_cols))

    @classmethod
    def tabular_data_from_csv(
        cls, csv_path: str, feature_cols: Union[str, List[str]], label_cols: Optional[Union[str, List[str]]] = None,
    ) -> SimpleDataloader:
        """Classmethod to create a dataset for tabular data from csv.

        Args:
            csv_path: The full path to csv.
            feature_cols: name(str) or list containing names feature columns.
            label_cols: name(str) or list containing names label columns.

        Returns:
            Tabular pytorch dataset.
        """
        return cls(TabularDataset.from_csv(csv_path=csv_path, feature_cols=feature_cols, label_cols=label_cols))

    @classmethod
    def text_data_from_df(
        cls, df: pd.DataFrame, input_col: str, label_cols: Optional[Union[str, List[str]]], tokenizer, max_len: int,
    ) -> SimpleDataloader:
        """Classmethod to create a dataset as required by transformers for text classification tasks.

        Args:
            df: The dataframe containing sentences and labels.
            input_col: The name of column containing sentences.
            label_cols: name of label column, or a list containing names of label columns.
            tokenizer: The tokenizer to be used to tokenize the sentences.
            max_len: The max_length to be used by the tokenizer.

        Returns:
                pytorch dataset for text classification using huggingface.
        """
        return cls(
            TextClassificationDataset.from_df(
                df=df, input_col=input_col, label_cols=label_cols, tokenizer=tokenizer, max_len=max_len,
            )
        )

    def get_loader(self, batch_size: int = 32, shuffle: bool = True, **dl_params) -> DataLoader:
        """Method to get dataloader.

        Args:
            batch_size(int): The batch size to use
            shuffle(bool): Whether to shuffle the inputs.
            **dl_params(dict) : Keyword arguments related to dataloader

        Returns:
            A PyTorch dataloader with given arguments.
        """
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle, **dl_params)
        return dl

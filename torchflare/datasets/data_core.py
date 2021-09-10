from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader

from torchflare.datasets.core_utils import (
    get_class_to_idx,
    get_iloc_cols,
    get_labels_from_paths,
    is_none,
)


class BaseDataset:
    """Base Dataset class."""

    def __init__(
        self,
        item_reader: "ItemReader",
        y: List = None,
        target_transforms: Optional[Callable] = None,
        **kwargs
    ):
        """Init Method.

        Args:
            item_reader: An object of type ItemReader.
            y: A list of targets.
            target_transforms: transforms to be applied on targets.
            **kwargs : Extra keyword arguments.
        """
        self.item_reader = item_reader
        self.y = y
        self.target_transforms = target_transforms
        self.is_y_none = is_none(self.y)
        self.input_transforms_fn = item_reader.apply_input_transforms
        self.target_transforms_fn = item_reader.apply_target_transforms

    def __len__(self):
        """__len__ method."""
        return len(self.item_reader.items)

    def __getitem__(self, idx: int):
        """Method to get item for particular index."""
        x = self.item_reader.get_item(idx)
        x = self.input_transforms_fn(transforms=self.item_reader.transforms, item=x)
        if not self.is_y_none:
            targets = self.target_transforms_fn(transforms=self.target_transforms, item=self.y[idx])
            return x, targets
        return x

    def batch(self, batch_size: int, shuffle: bool, **kwargs) -> DataLoader:
        """Method to create PyTorch style dataloaders.

        Args:
            batch_size: The batch size for the dataloader.
            shuffle : set to True to have the data reshuffled at every epoch.
            **kwargs : keyword arguments to be used by dataloader.
        """
        return DataLoader(self, batch_size, shuffle, **kwargs)


class ItemReader:
    """General Class to read input data and targets."""

    def __init__(
        self, items: List, transforms, df: pd.DataFrame = None, path: Path = None, **kwargs
    ):
        """Init Method.

        Args:
            items: list of items.
            transforms: transforms to apply on items.
            df: A pandas Dataframe
            path: A pathlib.Path object.
            **kwargs: extra keyword arguments.
        """
        self.items = items
        self.transforms = transforms
        self.df = df
        self.path = path
        self.base_dataset = BaseDataset

    def apply_input_transforms(self, transforms, item):
        """Method to apply transformations to inputs.

        Args:
            transforms : The transforms to be applied to input.
            item : The input.
        """
        raise NotImplementedError

    def apply_target_transforms(self, transforms, item):
        """Method to apply transformations to inputs.

        Args:
            transforms : The transforms to be applied to targets.
            item : The input.
        """
        raise NotImplementedError

    def get_item(self, idx):
        """Get item for particular index."""
        return self.items[idx]

    @classmethod
    def from_df(
        cls,
        df: DataFrame,
        path: Union[str, Path],
        input_columns: List[str],
        transforms=None,
        **kwargs
    ):
        """Method to read data from dataframes.

        Args:
            df : A pandas dataframe.
            path : The path where files are stored(to be used incase of image data,etc)
            input_columns : The columns which have input data.
            transforms : The transforms to be applied on the input.
        """
        raise NotImplementedError

    @classmethod
    def from_folders(cls, path, transforms=None, **kwargs):
        """Method to read data from folders.

        Args:
            path : The path to folder.
            transforms : The transforms to be applied to the input.
        """
        raise NotImplementedError

    @classmethod
    def from_csv(cls, csv_path, input_columns, transforms=None, path=None, **kwargs):
        """Method to read the data from csv.

        Args:
            csv_path: The full path to the csv.
            input_columns : The input columns.
            transforms: The transforms to be applied to inputs.
            path: The path to images(Used only in image classification).
        """
        raise NotImplementedError

    def _create_dataset(self, labels, target_transforms=None, **kwargs):
        return self.base_dataset(
            item_reader=self, y=labels, target_transforms=target_transforms, **kwargs
        )

    def targets_from_df(
        self, target_columns: List[str], target_transforms: Optional[Callable] = None, **kwargs
    ) -> "BaseDataset":
        """Method to read targets from dataframes.

        Args:
            target_columns: A list of target_columns.
            target_transforms : A callable to be applied to targets.
        """
        labels = get_iloc_cols(self.df, target_columns)
        return self._create_dataset(labels=labels, target_transforms=target_transforms, **kwargs)

    def targets_from_folders(
        self, target_path: Union[str, Path], target_transforms: Optional[Callable] = None, **kwargs
    ) -> "BaseDataset":
        """Method to read targets from folders.

        Args:
            target_path: The path to target folders.
            target_transforms : The transforms to be applied to targets.
        """
        labels_to_idx = get_labels_from_paths(target_path=target_path)
        labels = [get_class_to_idx(item=item, class_mapping=labels_to_idx) for item in self.items]

        return self._create_dataset(labels=labels, target_transforms=target_transforms, **kwargs)

    def add_test(self):
        """Method to be used create dataset for inference."""
        return self._create_dataset(labels=None, target_transforms=None)


__all__ = ["ItemReader", "BaseDataset"]

"""Wrapper for dataloaders."""
from __future__ import annotations

from typing import List, Union

import pandas as pd
from torch.utils.data import DataLoader

from torchflare.datasets.tabular import TabularDataset


class TabularDataloader:
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
        df: pd.DataFrame,
        feature_cols: Union[str, List[str]],
        label_cols: Union[str, List[str]] = None,
    ):
        """Classmethod to create dataset for tabular data from dataframe.

        Args:
            df: The dataframe containing features and labels.
            feature_cols: name(str) or list containing names feature columns.
            label_cols: name(str) or list containing names label columns.

        Returns:
            Tabular pytorch dataset.

        Examples:
            .. code-block:: python

                from torchflare.datasets import TabularDataloader

                dl = TabularDataloader.from_df(df=df,
                                            feature_cols= ["col1" , "col2"],
                                            label_cols="labels"
                                            ).get_loader(batch_size=64, # Required Args.
                                                           shuffle=True, # Required Args.
                                                           num_workers = 0, # keyword Args.
                                                           collate_fn = collate_fn # keyword Args.)

        """
        return cls(TabularDataset.from_df(df=df, feature_cols=feature_cols, label_cols=label_cols))

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        feature_cols: Union[str, List[str]],
        label_cols: Union[str, List[str]] = None,
    ):
        """Classmethod to create a dataset for tabular data from csv.

        Args:
            csv_path: The full path to csv.
            feature_cols: name(str) or list containing names feature columns.
            label_cols: name(str) or list containing names label columns.

        Returns:
            Tabular pytorch dataset.

        Examples:

            .. code-block:: python

                from torchflare.datasets import TabularDataloader
                dl = TabularDataloader.from_csv(csv_path="/train/train_data.csv",
                                feature_cols=["col1" , "col2"],
                                label_cols="labels"
                                ).get_loader(batch_size=64, # Required Args.
                                           shuffle=True, # Required Args.
                                           num_workers = 0, # keyword Args.
                                           collate_fn = collate_fn # keyword Args.)

        """
        return cls(TabularDataset.from_csv(csv_path=csv_path, feature_cols=feature_cols, label_cols=label_cols))

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


__all__ = ["TabularDataloader"]

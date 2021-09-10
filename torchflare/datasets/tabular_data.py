import pathlib
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from pandas import DataFrame

from torchflare.datasets.core_utils import get_iloc_cols, to_tensor
from torchflare.datasets.data_core import ItemReader


class TabularDataset(ItemReader):
    """PyTorch style datasets for Tabular-data."""

    def apply_input_transforms(self, transforms: Callable, item: Any):
        """Apply transforms to inputs."""
        if transforms is not None:
            return transforms(item)
        return to_tensor(item)

    def apply_target_transforms(self, transforms: Callable, item: Any):
        """Apply transforms to targets."""
        if transforms is not None:
            return transforms(item)
        return to_tensor(item)

    # skipcq : PYL-W0221
    @classmethod
    def from_df(
        cls,
        df: DataFrame,
        input_columns: List[str],
        transforms: Optional[Callable] = None,
        **kwargs
    ):
        """Classmethod to create pytorch style dataset from dataframes.

        Args:
            df: The dataframe which has inputs, and the labels/targets.
            input_columns: A list containing name of input columns.
            transforms : A callable which applies transforms on input data.

        Example:

            .. code-block::

                from torchflare.datasets import TabularDataset
                ds = TabularDataset.from_df(df=df,
                        feature_cols=["col1", "col2"]
                    ).targets_from_df(target_columns=["labels"])
        """
        items = get_iloc_cols(df, input_columns)
        return cls(items=items, df=df, transforms=transforms, path=None, **kwargs)

    # skipcq : PYL-W0221
    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, pathlib.Path],
        input_columns: List[str],
        transforms: Optional[Callable] = None,
        **kwargs
    ):
        """Classmethod to create pytorch style dataset from csv file.

        Args:
            csv_path: The full path to csv.
            input_columns: A list containing name of input columns.
            transforms : A callable which applies transforms on input data.

        Example:

            .. code-block:: python

                from torchflare.datasets import TabularDataset

                ds = TabularDataset.from_csv(
                    csv_path="/train/train_data.csv", feature_cols=["col1", "col2"]
                    ).targets_from_df(target_columns=["labels"])
        """
        df = pd.read_csv(csv_path)
        return cls.from_df(df=df, input_columns=input_columns, **kwargs)


__all__ = ["TabularDataset"]

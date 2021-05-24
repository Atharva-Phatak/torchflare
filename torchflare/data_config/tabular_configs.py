"""Implements Tabular Data configs."""
from dataclasses import MISSING, dataclass, field
from typing import List, Union

import pandas as pd

from torchflare.data_config.base import BaseConfig
from torchflare.datasets.tabular import TabularDataset


@dataclass
class _TabularDataConfigDF:
    """Class to create data config for tabular data."""

    df: pd.DataFrame = field(
        default=MISSING, metadata={"help": "The dataframe which has inputs, and the labels/targets."}
    )
    feature_cols: Union[str, List[str]] = field(
        default=MISSING,
        metadata={
            "help": "The name of columns which contain inputs. \
                feature_cols can be a string if single column or can be a list of string if multiple columns."
        },
    )
    label_cols: Union[str, List[str]] = field(
        default=None,
        metadata={
            "help": "The name of columns which contain the labels. \
                label_cols can be a string or can be a list of string if multiple columns are used."
        },
    )


@dataclass
class _TabularDataConfigCSV:
    """Class to create data config for tabular data."""

    csv_path: str = field(default=MISSING, metadata={"help": "The full path to csv."})
    feature_cols: Union[str, List[str]] = field(
        default=MISSING,
        metadata={
            "help": "The name of columns which contain inputs. \
                   feature_cols can be a string if single column or can be a list of string if multiple columns."
        },
    )
    label_cols: Union[str, List[str]] = field(
        default=None,
        metadata={
            "help": "The name of columns which contain the labels. \
                   label_cols can be a string or can be a list of string if multiple columns are used."
        },
    )


class TabularDataConfig(BaseConfig):
    """Class to create data configs for tabular data."""

    def __init__(self, config, data_method):
        """Constructor Method.

        Args:
            config: The config object.
            data_method: The method which will be used to create the dataset.
        """
        super(TabularDataConfig, self).__init__(config=config, data_method=data_method)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        feature_cols: Union[str, List[str]],
        label_cols: Union[str, List[str]] = None,
    ):
        """Classmethod to create pytorch style dataset from dataframes.

        Args:
            df: The dataframe which has inputs, and the labels/targets.
            feature_cols: The name of columns which contain inputs.
                   feature_cols can be a string if single column or can be a list of string if multiple columns.
            label_cols: The name of columns which contain the labels.
                   label_cols can be a string or can be a list of string if multiple columns are used.

        Returns:
               returns config object and a method to create a Pytorch-style dataset.
        """
        return cls(
            _TabularDataConfigDF(df=df, feature_cols=feature_cols, label_cols=label_cols),
            data_method=TabularDataset.from_df,
        )

    @classmethod
    def from_csv(cls, csv_path, feature_cols: Union[str, List[str]], label_cols: Union[str, List[str]] = None):
        """Classmethod to create pytorch style dataset from csv file.

        Args:
            csv_path: The full path to csv.
            feature_cols: The name of columns which contain inputs.
                feature_cols can be a string if single column or can be a list of string if multiple columns.
            label_cols: The name of columns which contain the labels.
                label_cols can be a string or can be a list of string if multiple columns are used.

        Returns:
             returns config object and a method to create a Pytorch-style dataset.
        """
        return cls(
            _TabularDataConfigCSV(csv_path=csv_path, feature_cols=feature_cols, label_cols=label_cols),
            data_method=TabularDataset.from_csv,
        )


__all__ = ["TabularDataConfig"]

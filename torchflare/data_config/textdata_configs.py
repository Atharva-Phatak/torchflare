"""Implements Tabular Data configs."""
from dataclasses import MISSING, dataclass, field
from typing import Any, List, Union

import pandas as pd

from torchflare.data_config.base import BaseConfig
from torchflare.datasets.text_dataset import TextClassificationDataset


@dataclass
class _TextDataConfigDF:
    """Class to create data config for text data."""

    df: pd.DataFrame = field(
        default=MISSING, metadata={"help": "The dataframe which has the input sentences and targets."}
    )
    input_col: str = field(default=MISSING, metadata={"help": "The column containing the inputs."})
    tokenizer: Any = field(
        default=MISSING, metadata={"help": " The tokenizer to be used.(Use only tokenizer available in huggingface."}
    )
    max_len: int = field(default=MISSING, metadata={"help": "The max_len to be used."})
    label_cols: Union[str, List[str]] = field(
        default=MISSING, metadata={"help": "The column which contains corresponding labels."}
    )


class TextDataConfig(BaseConfig):
    """Class to create Text data configs for text classification/Regression."""

    def __init__(self, config, data_method):
        super(TextDataConfig, self).__init__(config=config, data_method=data_method)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        input_col: str,
        tokenizer,
        max_len: int,
        label_cols: Union[str, List[str]] = None,
    ):
        """Classmethod to create the dataset from dataframe.

        Args:
            df: The dataframe which has the input sentences and targets.
            input_col: The column containing the inputs.
            label_cols: The column which contains corresponding labels.
            tokenizer: The tokenizer to be used.(Use only tokenizer available in huggingface.
            max_len: The max_len to be used.

        Returns:
            returns config object and a method to create a Pytorch-style dataset.
        """
        return cls(
            config=_TextDataConfigDF(
                df=df, input_col=input_col, tokenizer=tokenizer, max_len=max_len, label_cols=label_cols
            ),
            data_method=TextClassificationDataset.from_df,
        )


__all__ = ["TextDataConfig"]

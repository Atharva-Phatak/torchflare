import pathlib
from typing import List, Union

import pandas as pd
from pandas import DataFrame

from torchflare.datasets.core_utils import get_iloc_cols, to_tensor
from torchflare.datasets.data_core import ItemReader


class TextDataset(ItemReader):
    """Class for text data as required by transformers."""

    def __init__(self, tokenizer, max_len, **kwargs):
        super(TextDataset, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def apply_input_transforms(self, transforms, item):
        """Method to apply input transforms to the inputs."""
        inps = self.tokenizer(
            item,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inps_dict = {
            k: inps[k].squeeze(0) for k in inps
        }  # Understand why extra dim is added by tokenizer
        return inps_dict

    def apply_target_transforms(self, transforms, item):
        """Method to apply target transforms to the targets."""
        if transforms is not None:
            return transforms(item)
        return to_tensor(item)

    # skipcq : PYL-W0221
    @classmethod
    def from_df(
        cls, df: DataFrame, input_columns: List[str], tokenizer=None, max_len=None, **kwargs
    ):
        """Classmethod to create the dataset from dataframe.

        Args:
            df: The dataframe which has the input sentences and targets.
            input_columns: A list containing names of input columns.
            tokenizer: The tokenizer to be used.(Use only tokenizer available in huggingface.
            max_len(int): The max_len to be used.

        Example:

            .. code-block:: python

                import transformers
                from torchflare.datasets import TextClassificationDataset

                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

                ds = TextClassificationDataset.from_df(
                    df=df, input_col=["tweet"], tokenizer=tokenizer, max_len=128
                    ).targets_from_df(target_columns=["label"])
        """
        items = get_iloc_cols(df, input_columns)
        return cls(
            items=items,
            path=None,
            transforms=None,
            tokenizer=tokenizer,
            max_len=max_len,
            df=df,
            **kwargs
        )

    # skipcq : PYL-W0221
    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, pathlib.Path],
        input_columns: List[str],
        tokenizer=None,
        max_len=None,
        **kwargs
    ):
        """Classmethod to create the dataset from dataframe.

        Args:
            csv_path : The full path to csv.
            input_columns: A list containing names of inputs columns.
            tokenizer: The tokenizer to be used.(Use only tokenizer available in huggingface.
            max_len(int): The max_len to be used.

        Example:

            .. code-block:: python

                import transformers
                from torchflare.datasets import TextClassificationDataset

                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

                ds = TextClassificationDataset.from_csv(
                    csv_path="/train/train.csv", input_col="tweet", tokenizer=tokenizer, max_len=128
                    ).targets_from_df(target_columns=["label"])
        """
        df = pd.read_csv(csv_path)
        return cls.from_df(
            df=df, input_columns=input_columns, tokenizer=tokenizer, max_len=max_len, **kwargs
        )


__all__ = ["TextDataset"]

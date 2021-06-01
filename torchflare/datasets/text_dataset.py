"""Implements dataset for text classification task."""
from __future__ import annotations

from typing import List, Union

import pandas as pd
from torch.utils.data import Dataset

from torchflare.datasets.utils import to_tensor


class TextClassificationDataset(Dataset):
    """Class to create a dataset for text classification as required by transformers."""

    def __init__(
        self,
        inputs: List[str],
        tokenizer,
        max_len: int,
        labels: [Union[str, List[str]]] = None,
    ):
        """Constructor Method.

        Args:
            inputs: A list of sentences.
            labels: A list of labels for classification.
            tokenizer: The huggingface tokenizer to be used.
            max_len : The max_length of ids.
        """
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """__len__ method.

        Returns:
            The length of dataset.
        """
        return len(self.inputs)

    def __getitem__(self, item) -> Union:
        """__getitem__ method.

        Args:
            item : idx

        Returns:
            A dictionary containing input_ids , token_type_ids ,etc and the corresponding labels.
        """
        text = self.inputs[item]
        inps = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inps_dict = {k: inps[k].squeeze(0) for k in inps}  # Understand why extra dim is added by tokenizer

        if self.labels is not None:
            labels = self.labels[item]
            return inps_dict, to_tensor(labels)

        return inps_dict

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        input_col: str,
        tokenizer,
        max_len: int,
        label_cols: Union[str, List[str]] = None,
    ) -> TextClassificationDataset:
        """Classmethod to create the dataset from dataframe.

        Args:
            df(pd.DataFrame): The dataframe which has the input sentences and targets.
            input_col(str): The column containing the inputs.
            label_cols(str or List(str): The column which contains corresponding labels.
            tokenizer: The tokenizer to be used.(Use only tokenizer available in huggingface.
            max_len(int): The max_len to be used.

        Returns:
            A list of sentences and corresponding labels if label_cols is provided else return a list of sentences.

        Examples:

            .. code-block:: python

                import transformers
                from torchflare.datasets import TextClassificationDataset

                tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

                ds = TextClassificationDataset.from_df(
                    df=df, input_col="tweet", label_cols="label", tokenizer=tokenizer, max_len=128)
        """
        inputs = df.loc[:, input_col].values.tolist()

        labels = df.loc[:, label_cols].values.tolist() if label_cols is not None else None

        return cls(inputs=inputs, labels=labels, tokenizer=tokenizer, max_len=max_len)


__all__ = ["TextClassificationDataset"]

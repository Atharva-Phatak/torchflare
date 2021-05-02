"""Wrapper for dataloaders."""
from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd
from torch.utils.data import DataLoader

from torchflare.datasets.text_dataset import TextClassificationDataset


class TextDataloader:
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
        input_col: str,
        label_cols: Optional[Union[str, List[str]]],
        tokenizer,
        max_len: int,
    ) -> TextDataloader:
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
                df=df,
                input_col=input_col,
                label_cols=label_cols,
                tokenizer=tokenizer,
                max_len=max_len,
            )
        )

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


__all__ = ["TextDataloader"]

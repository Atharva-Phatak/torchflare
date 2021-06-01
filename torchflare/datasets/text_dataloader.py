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
            df(pd.DataFrame): The dataframe which has the input sentences and targets.
            input_col(str): The column containing the inputs.
            label_cols(str or List(str): The column which contains corresponding labels.
            tokenizer: The tokenizer to be used.(Use only tokenizer available in huggingface.
            max_len(int): The max_len to be used.

        Returns:
                pytorch dataset for text classification using huggingface.

        Examples:

            .. code-block:: python

                from torchflare.datasets import TextDataloader

                dl = TextDataloader.data_from_df(df=df,
                                  input_col="tweet",
                                  label_cols="label",
                                  tokenizer=tokenizer,
                                   max_len=128
                                   ).get_loader(batch_size=64, # Required Args.
                                                       shuffle=True, # Required Args.
                                                       num_workers = 0, # keyword Args.
                                                       collate_fn = collate_fn # keyword Args.
                                                       )


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

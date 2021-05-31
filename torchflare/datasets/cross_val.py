"""Utility for cross_validation."""
from typing import Dict, Tuple

import sklearn
import torch
from torch.utils.data import DataLoader, Dataset

from torchflare.datasets.image_classification import ImageDataset
from torchflare.datasets.segmentation import SegmentationDataset
from torchflare.datasets.tabular import TabularDataset
from torchflare.datasets.text_dataset import TextClassificationDataset


class CVSplit:
    """Class to perform cross validation on given dataset.

    Args:
        dataset: A PyTorch style dataset. Dataset must be the one implemented in torchflare.
        cv: The cross-validation splitting strategy.
        n_splits: The number of splits.
        **kwargs: keyword arguments related to cross validation strategy.

    Note:
        Only supports KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, RepeatedKFold,
        RepeatedStratifiedKFold cross validation schemes.

    Raises:
        ValueError if cv strategy not in the specified ones.
    """

    def __init__(self, dataset: Dataset, cv: str, n_splits: int, **kwargs):
        """Constructor class for CVSplit class.

        Args:
            dataset: A PyTorch style dataset. Dataset must be the one implemented in torchflare.
            cv: The cross-validation splitting strategy.
            n_splits: The number of splits.
            **kwargs: keyword arguments related to cross validation strategy.

        Note:
            Only supports KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, RepeatedKFold,
            RepeatedStratifiedKFold cross validation schemes.

        Raises:
            ValueError if cv strategy not in the specified ones.

        """
        if cv not in [
            "KFold",
            "ShuffleSplit",
            "StratifiedKFold",
            "StratifiedShuffleSplit",
            "RepeatedKFold",
            "RepeatedStratifiedKFold",
        ]:
            raise ValueError(f"Does not support {cv}")
        self.X, self.y = None, None
        self.cv = getattr(sklearn.model_selection, cv)(n_splits=n_splits, **kwargs)
        self.dataset = dataset
        self.fold_dict = {}
        self._get_fold()

    def _get_inputs(self):
        """Creates X,y for cross validation depending upon dataset."""
        if isinstance(self.dataset, (ImageDataset, TabularDataset, TextClassificationDataset)):
            self.X, self.y = self.dataset.inputs, self.dataset.labels

        elif isinstance(self.dataset, SegmentationDataset):
            raise ValueError("Segmentation data is not supported")

    def _get_fold(self):
        """Generated fold dictionary for given cross validation scheme."""
        self._get_inputs()
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X, self.y)):
            self.fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    def get_loaders(self, fold: int, train_params: Dict, val_params: Dict) -> Tuple[DataLoader, DataLoader]:
        """Generates training and validation dataloaders as per the given fold.

        Args:
            fold: The fold/split number for which you want dataloader.
            train_params: A dictionary containing parameters for training dataloader.
            val_params: A dictionary containing parameters for validation dataloader.

        Returns:
            Training dataloader and validation dataloader for a given fold.
        """
        train_data = torch.utils.data.Subset(self.dataset, self.fold_dict[fold].get("train_idx"))
        valid_data = torch.utils.data.Subset(self.dataset, self.fold_dict[fold].get("val_idx"))
        train_dl = torch.utils.data.DataLoader(train_data, **train_params)
        valid_dl = torch.utils.data.DataLoader(valid_data, **val_params)

        return train_dl, valid_dl

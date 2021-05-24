"""Implements Base classes."""
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable


def _validate_config(cfg):
    if not is_dataclass(cfg):
        raise ValueError("The data config must be a dataclass.")


class BaseConfig:
    """Base class for creating other data config classes."""

    def __init__(self, config: dataclass, data_method: Callable):
        """Constructor method.

        Args:
            config: configurations dataclass.
            data_method: The method which will be used to create a pytorch-style dataset.
        """
        _validate_config(config)
        self.config = asdict(config)
        self.data_method = data_method


class DataPipe:
    """Base Pipeline to create datasets out of data config objects."""

    def __init__(self, train_data_cfg, valid_data_cfg):
        """Constructor Method.

        Args:
            train_data_cfg: The training data config.
            valid_data_cfg: The validation data config.
        """
        self.train_ds = self._create_dataset(data_cfg=train_data_cfg)
        self.valid_ds = self._create_dataset(data_cfg=valid_data_cfg)

    @staticmethod
    def _create_dataset(data_cfg):
        return data_cfg.data_method(**data_cfg.config)

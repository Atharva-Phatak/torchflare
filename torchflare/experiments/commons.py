from enum import Enum


class EVENTS(Enum):
    """Events that are used by experiment class."""

    ON_EXPERIMENT_START = "on_experiment_start"
    ON_EXPERIMENT_END = "on_experiment_end"
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_LOADER_START = "on_loader_start"
    ON_LOADER_END = "on_loader_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"


TRAIN_ATTRS = ["nn_module", "optimizer", "criterion"]
ATTR_TO_INTERNAL = {"nn_module": "model", "optimizer": "optimizer", "criterion": "criterion"}

__all__ = ["TRAIN_ATTRS", "ATTR_TO_INTERNAL", "EVENTS"]

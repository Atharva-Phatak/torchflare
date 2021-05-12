"""Definitions of experiment states and Callback order."""
from enum import Enum, IntEnum


class ExperimentStates(Enum):
    """Class Define various stages of Training."""

    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_EXP_START = "on_experiment_start"
    ON_EXP_END = "on_experiment_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_LOADER_START = "on_loader_start"
    ON_LOADER_END = "on_loader_end"
    STOP = "stop"


class CallbackOrder(IntEnum):
    """Callback orders."""

    INTERNAL = 0
    LOGGING = 1
    STOPPING = 2
    EXTERNAL = 3


__all__ = ["ExperimentStates", "CallbackOrder"]

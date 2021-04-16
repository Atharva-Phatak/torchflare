"""Definitions of experiment states and Callback order."""
from enum import Enum, IntEnum


class ExperimentStates(Enum):
    """Class Define various stages of Training."""

    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    EXP_START = "experiment_start"
    EXP_END = "experiment_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    STOP = "stop"


class CallbackOrder(IntEnum):
    """Callback orders."""

    INTERNAL = 0
    LOGGING = 1
    STOPPING = 2
    EXTERNAL = 3


__all__ = ["ExperimentStates", "CallbackOrder"]

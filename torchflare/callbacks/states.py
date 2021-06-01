"""Definitions of experiment states and Callback order."""
from enum import IntEnum


class CallbackOrder(IntEnum):
    """Callback orders."""

    MODEL_INIT = 0
    LOSS = 1
    METRICS = 2
    SCHEDULER = 3
    CHECKPOINT = 4
    LOGGING = 5
    STOPPING = 6
    EXTERNAL = 7


__all__ = ["CallbackOrder"]

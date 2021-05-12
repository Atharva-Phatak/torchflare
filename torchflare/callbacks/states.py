"""Definitions of experiment states and Callback order."""
from enum import IntEnum


class CallbackOrder(IntEnum):
    """Callback orders."""

    INTERNAL = 0
    LOGGING = 1
    STOPPING = 2
    EXTERNAL = 3


__all__ = ["CallbackOrder"]

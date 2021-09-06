import contextlib

import torch


# noinspection PyMethodMayBeStatic
class BaseBackend:
    """Class to perform standard steps for optimizer, backward loss etc."""

    def __init__(self):
        self.autocast = contextlib.nullcontext()

    # skipcq :  PYL-R1705
    def zero_grad(self, optimizer) -> None:
        """Wrapper for optimizer.zero_grad()."""
        optimizer.zero_grad()

    # skipcq :  PYL-R1705
    def backward_loss(self, loss) -> None:
        """Method to propogate loss backward."""
        # skipcq: PYL-W0106
        loss.backward()

    # skipcq :  PYL-R1705
    def optimizer_step(self, optimizer) -> None:
        """Method to perform optimizer step."""
        optimizer.step()


# noinspection PyMethodMayBeStatic
class AMPBackend:
    """Class to perform standard steps for optimizer , scaling using mixed precision."""

    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast = torch.cuda.amp.autocast()

    # skipcq :  PYL-R1705
    def zero_grad(self, optimizer) -> None:
        """Wrapper for optimizer.zero_grad()."""
        optimizer.zero_grad()

    def backward_loss(self, loss) -> None:
        """Method to propogate loss backward."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, optimizer) -> None:
        """Method to perform optimizer step."""
        self.scaler.step(optimizer)
        self.scaler.update()


__all__ = ["BaseBackend", "AMPBackend"]

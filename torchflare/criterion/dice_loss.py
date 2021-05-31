"""Implements DiceLoss."""
import torch
import torch.nn as nn

from torchflare.metrics.dice_meter import DiceScore


class DiceLoss(nn.Module):
    """Implementation of Dice Loss.

    Args:
            class_dim: The dimension indication class.
    """

    def __init__(self, class_dim=1):
        """Constructor method for Dice Loss."""
        super(DiceLoss, self).__init__()
        self.dice = DiceScore(threshold=None, class_dim=class_dim)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            outputs: outputs from the net after applying activations.
            targets: The targets.

        Returns:
            The computed loss value.
        """
        self.dice.reset()
        self.dice.accumulate(outputs=outputs, targets=targets)
        return 1 - self.dice.value


__all__ = ["DiceLoss"]

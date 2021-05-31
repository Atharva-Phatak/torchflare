"""Implements IouLoss."""
import torch
import torch.nn as nn

from torchflare.metrics.iou_meter import IOU


class IOULoss(nn.Module):
    """Computes intersection over union Loss.

    IOULoss =  1 - iou_score

    Args:
            class_dim: indicates class dimension (K) for
                outputs and targets tensors (default = 1)
    """

    def __init__(self, class_dim=1):
        """Constructor method for IOULoss."""
        super(IOULoss, self).__init__()
        self.iou = IOU(threshold=None, class_dim=class_dim)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward Method.

        Args:
            outputs: outputs from the net after applying activations.
            targets: The targets.

        Returns:
            The computed loss value.
        """
        self.iou.reset()
        self.iou.accumulate(outputs=outputs, targets=targets)
        return 1 - self.iou.value


__all__ = ["IOULoss"]

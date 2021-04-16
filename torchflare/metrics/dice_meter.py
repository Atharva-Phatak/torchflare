"""Implements Dice Score."""
import torch

from torchflare.metrics.meters import MetricMeter, calculate_segmentation_statistics


# To-Do better implementation for Dice
class DiceScore(MetricMeter):
    """Class to compute Dice Score."""

    def __init__(self, threshold: float = None, class_dim: int = 1):
        """Constructor method for DiceScore.

        Args:
            threshold: threshold for binarization of predictions
            class_dim: indicates class dimension (K)

        Note:
             Supports only binary cases
        """
        self.threshold = threshold
        self.class_dim = class_dim
        self.eps = 1e-20

        self._outputs = []
        self._targets = []
        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Class to accumulate the outputs and targets.

        Args:
            outputs: [N, K, ...] tensor that for each of the N samples
                indicates the probability of the sample belonging to each of
                the K num_classes.
            targets:  binary [N, K, ...] tensor that encodes which of the K
                num_classes are associated with the N-th sample.
        """
        self._outputs.append(outputs)
        self._targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Computes the dice score.

        Returns:
            The computed Dice score.
        """
        self._outputs = torch.cat(self._outputs)
        self._targets = torch.cat(self._targets)

        tp, fp, fn = calculate_segmentation_statistics(
            outputs=self._outputs, targets=self._targets, threshold=self.threshold, class_dim=self.class_dim,
        )

        union = tp + fp + fn

        score = (2 * tp + self.eps * (union == 0).float()) / (2 * tp + fp + fn + self.eps)

        return torch.mean(score)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []


__all__ = ["DiceScore"]

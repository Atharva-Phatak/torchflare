"""Implements Recall-score."""
import torch

from torchflare.metrics.meters import MetricMeter, _BaseInputHandler


class Recall(_BaseInputHandler, MetricMeter):
    """Class to compute Recall Score.

    Support binary, multiclass and multilabel cases
    """

    def __init__(
        self, num_classes: int, average: str = "macro", threshold: float = 0.5, multilabel: bool = False,
    ):
        """Constructor method for Precision Class.

        Args:
            num_classes: The number of num_classes.
            average: The type of reduction to apply.
                macro: calculate metrics for each class and averages them with equal weightage to each class.
                micro: calculate metrics globally for each sample and class.
            threshold: The threshold value to transform probability predictions to binary values(0,1)
            multilabel: Set it to True if your problem is  multilabel classification.
        """
        super(Recall, self).__init__(
            num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,
        )

        self._outputs = None
        self._targets = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The name of the class
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Accumulates the batch outputs and targets.

        Args:
            outputs : raw logits from the network.
            targets : targets to use for computing accuracy
        """
        outputs, targets = self.detach_tensor(outputs), self.detach_tensor(targets)
        self._outputs.append(outputs)
        self._targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Compute the recall score.

        Returns:
            The computed recall score.
        """
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)

        tp, fp, tn, fn = self.compute_stats(outputs=outputs, targets=targets)
        recall = self.reduce(numerator=tp, denominator=tp + fn)
        return recall

    def reset(self):
        """Reset the output and target lists."""
        self._outputs = []
        self._targets = []


__all__ = ["Recall"]

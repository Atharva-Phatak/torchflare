"""Implements FBeta and F1-score."""
import torch

from torchflare.metrics.meters import MetricMeter, _BaseInputHandler


class FBeta(_BaseInputHandler, MetricMeter):
    """Computes Fbeta Score.

    Supports binary,multiclass and multilabel cases.

    Args:
        num_classes(int) : The number of num_classes(For binary case , use out_features : 1)
        threshold(float): The value of threshold for masking. Input is raw logits.
        average(str): One of "micro" or "macro"
        beta(float): weight of precision in harmonic mean.
        multilabel(bool): Whether problem is multilabel or not.

    Examples:

        .. code-block:: python

        from torchflare.metrics import FBeta

        # Binary-Classification Problems
        acc = FBeta(num_classes=2, threshold=0.7, multilabel=False, average="macro")

        # Mutliclass-Classification Problems
        multiclass_acc = FBeta(num_classes=4, multilabel=False, average="macro")

        # Multilabel-Classification Problems
        multilabel_acc = FBeta(num_classes=5, multilabel=True, threshold=0.7, average="macro")
    """

    def __init__(
        self,
        beta: float,
        num_classes: int,
        threshold: float = 0.5,
        average: str = "macro",
        multilabel: bool = False,
    ):
        """Constructor method for Fbeta score."""
        super(FBeta, self).__init__(
            num_classes=num_classes,
            multilabel=multilabel,
            threshold=threshold,
            average=average,
        )

        self.beta = beta
        self.eps = 1e-20

        self._outputs = None
        self._targets = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to accumulate the outputs and targets.

        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : Ground truth targets
        """
        outputs, targets = self.detach_tensor(outputs), self.detach_tensor(targets)

        self._outputs.append(outputs)
        self._targets.append(targets)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []

    @property
    def value(self) -> torch.Tensor:
        """Computes the FBeta Score.

        Returns:
            The computed Fbeta score.
        """
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)

        tp, fp, tn, fn = self.compute_stats(outputs=outputs, targets=targets)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        numerator = (1 + self.beta ** 2) * precision * recall
        denominator = self.beta ** 2 * precision + recall

        fbeta = self.reduce(numerator=numerator, denominator=denominator)
        return fbeta


class F1Score(_BaseInputHandler, MetricMeter):
    """Computes F1 Score.

    Supports binary,multiclass and multilabel cases.

    Args:
            num_classes : The number of num_classes(For binary case , use out_features : 1)
            threshold: The value of threshold for masking. Input is raw logits.
            average : One of "micro" or "macro".
            multilabel: Whether the problem is multilabel or not.

    Examples:
        .. code-block:: python

            from torchflare.metrics import F1Score

            # Binary-Classification Problems
            acc = F1Score(num_classes=2, threshold=0.7, multilabel=False, average="macro")

            # Mutliclass-Classification Problems
            multiclass_acc = F1Score(num_classes=4, multilabel=False, average="macro")

            # Multilabel-Classification Problems
            multilabel_acc = F1Score(num_classes=5, multilabel=True, threshold=0.7, average="macro")
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        multilabel: bool = False,
        average: str = "macro",
    ):
        """Constructor method for F1-score."""
        super(F1Score, self).__init__(
            num_classes=num_classes,
            multilabel=multilabel,
            threshold=threshold,
            average=average,
        )

        self.eps = 1e-20
        self._outputs = None
        self._targets = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    @property
    def value(self) -> torch.Tensor:
        """Value of FBeta Score.

        Returns:
            The computed F1-score
        """
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)

        tp, fp, tn, fn = self.compute_stats(outputs=outputs, targets=targets)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        numerator = 2 * precision * recall
        denominator = precision + recall

        f1 = self.reduce(numerator=numerator, denominator=denominator)
        return f1

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to accumulate the outputs and targets.

        Args:
            outputs : raw logits from the network.
            targets : Ground truth targets
        """
        outputs, targets = self.detach_tensor(outputs), self.detach_tensor(targets)

        self._outputs.append(outputs)
        self._targets.append(targets)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []


__all__ = ["FBeta", "F1Score"]

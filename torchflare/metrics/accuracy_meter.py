"""Implements Accuracy-metric."""
import torch

from torchflare.metrics.meters import MetricMeter, _BaseMetric


class Accuracy(_BaseMetric, MetricMeter):
    """Computes Accuracy.

    Support binary,multilabel and multiclass cases
    """

    def __init__(self, num_classes: int, threshold: float = 0.5, multilabel: bool = False):
        """Constructor method for Accuracy Class.

        Args:
            num_classes: The number of num_classes.
            threshold: The threshold value to transform probability predictions to binary values(0,1)
            multilabel: Set it to True if your problem is  multilabel classification.
        """
        super(Accuracy, self).__init__(multilabel=multilabel)

        self.threshold = threshold
        self.num_classes = num_classes
        self._outputs = None
        self._targets = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    # noinspection PyTypeChecker
    def _compute(self, outputs: torch.Tensor, targets: torch.Tensor):

        self._check_type(outputs=outputs, targets=targets)
        if self.case_type == "multiclass":

            outputs = torch.argmax(outputs, dim=1)
            correct = torch.eq(outputs, targets).view(-1)

        else:

            outputs = torch.sigmoid(outputs)
            outputs = (outputs >= self.threshold).float()

            if self.case_type == "binary":
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                correct = torch.eq(outputs, targets)
            else:
                last_dim = outputs.ndimension()
                outputs = torch.transpose(outputs, 1, last_dim - 1).reshape(-1, self.num_classes)
                targets = torch.transpose(targets, 1, last_dim - 1).reshape(-1, self.num_classes)
                correct = torch.all(targets == outputs.type_as(targets), dim=-1)
        # assert outputs.shape == targets.shape
        total = correct.shape[0]
        correct = torch.sum(correct)
        return correct, total

    def accumulate(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to accumulate the outputs and targets.

        Args:
            outputs : raw logits from the network.
            targets : Ground truth targets
        """
        outputs, targets = self.detach_tensor(outputs), self.detach_tensor(targets)
        self._outputs.append(outputs)
        self._targets.append(targets)

    def compute(self) -> torch.Tensor:
        """Computes the Accuracy per epoch.

        Returns:
            The accuracy
        """
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)
        correct, total = self._compute(outputs=outputs, targets=targets)
        return correct / total

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []


__all__ = ["Accuracy"]

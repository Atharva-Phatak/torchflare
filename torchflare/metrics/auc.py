from abc import ABC

import torch

from torchflare.metrics.meters import MetricMeter


def _auc_check(x, y):
    if x.ndim > 1:
        x = x.view(-1)

    if y.ndim > 1:
        y = y.view(-1)
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both x and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in x and y tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _detach_tensor(x) -> torch.Tensor:
    """Moves tensor to cpu from gpu.

    Returns:
        Torch.Tensor
    """
    return x.detach().cpu()


# Source/Adapted: https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_ranking.py#L826
class AUC(MetricMeter, ABC):
    """Computes Area Under the Curve (AUC) using the trapezoidal rule.

    Forward accepts two input tensors that should be 1D and have the same number
    of elements.
    """

    def __init__(self):
        """Constructor Method for AUC Score."""
        super(AUC, self).__init__()
        self._outputs = None
        self._targets = None
        self.reset()

    def accumulate(self, outputs, targets):
        """Method to accumulate the outputs and targets.

        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : Ground truth targets
        """
        outputs, targets = _detach_tensor(outputs), _detach_tensor(targets)
        outputs, targets = _auc_check(x=outputs, y=targets)
        self._outputs.append(outputs)
        self._targets.append(targets)

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    @staticmethod
    def _compute(x, y):

        x, x_ids = torch.sort(x)
        y = y[x_ids]

        dx = x[1:] - x[:-1]
        if (dx < 0).any():

            if (dx < 0).all():
                direction = -1
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`."
                )
        else:
            direction = 1.0

        return direction * torch.trapz(y, x)

    @property
    def value(self):
        """Computes the AUC score.

        Returns:
            The computed Dice score.
        """
        outputs = torch.cat(self._outputs)
        targets = torch.cat(self._targets)
        return self._compute(x=outputs, y=targets)

    def reset(self):
        """Resets the accumulation lists."""
        self._outputs = []
        self._targets = []

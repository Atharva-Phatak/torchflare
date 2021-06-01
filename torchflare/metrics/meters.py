"""Implementation of utilities for metrics."""
from typing import Tuple

import torch
from einops import reduce


class _BaseMetric:
    """A Class which decides type of classification i.e. binary,multilabel or multiclass."""

    def __init__(self, multilabel: bool = False):
        """Constructor class for BaseMetric class.

        Args:
            multilabel(bool): Set to True if problem type is multilabel.
        """
        self.multilabel = multilabel
        self.case_type = None

    @staticmethod
    def _check_shape(outputs: torch.Tensor, targets: torch.Tensor):
        """Function to check if there is a mismatch between outputs and targets.

        Args:
            outputs(torch.Tensor): The outputs of the net.
            targets(torch.Tensor): The targets.

        Raises:
            ValueError:  If shapes does not match.
        """
        if not (outputs.ndim == targets.ndim or outputs.ndim == targets.ndim + 1):
            raise ValueError("Preds and Targets must have same number of dimensions")

    @staticmethod
    def _convert_to_onehot(num_classes: int, indices: torch.Tensor) -> torch.Tensor:
        """Converts tensor to one_hot representation.

        Args:
            num_classes(int): The number of classes.
            indices(torch.Tensor): torch.Tensor.

        Returns:
            one_hot converted tensor.
        """
        onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:], dtype=indices.dtype)
        index = indices.long().unsqueeze(1).expand_as(onehot)
        return onehot.scatter_(1, index, 1.0)

    @staticmethod
    def detach_tensor(x: torch.Tensor) -> torch.Tensor:
        """Detaches the tensor."""
        return x.detach().cpu()

    # noinspection PyUnboundLocalVariable
    def _check_type(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Method to infer type of the problem."""
        self._check_shape(outputs, targets)

        if targets.ndim + 1 == outputs.ndim:
            if outputs.shape[1] == 1:
                case_type = "binary"
            else:
                case_type = "multiclass"
        elif outputs.ndim == targets.ndim:
            if self.multilabel:
                case_type = "multilabel"
            else:
                case_type = "binary"

        if self.case_type is None:
            self.case_type = case_type


class _BaseInputHandler(_BaseMetric):
    """Class to handle shapes for various classification tasks."""

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        multilabel: bool = False,
        average: str = "macro",
    ):
        """Constructor method.

        Args:
            num_classes(int): The number of classes.
            threshold(float): The threshold for binarization.
            multilabel(bool): Whether the problem is multilabel or not.
            average(str): One of macro or micro.
        """
        super(_BaseInputHandler, self).__init__(multilabel=multilabel)
        self.num_classes = num_classes
        self.threshold = threshold
        self.multilabel = multilabel
        self.eps = 1e-20
        self.average = average
        assert self.average in ["micro", "macro"], "Average should be one of ['micro , 'macro'] "  # noqa: S101

    @staticmethod
    def _calculate_stats(
        true_preds: torch.Tensor,
        false_preds: torch.Tensor,
        pos_preds: torch.Tensor,
        neg_preds: torch.Tensor,
    ):
        tp = true_preds * pos_preds
        fp = false_preds * pos_preds
        tn = true_preds * neg_preds
        fn = false_preds * neg_preds

        return tp, fp, tn, fn

    def compute_stats(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Computes true_positives, false_positives, true_negatives, false_negatives.

        Args:
            outputs(torch.Tensor): The outputs of the net.
            targets(torch.Tensor): The targets.

        Returns:
            True positives , false positives, true negatives , false negatives.
        """
        outputs, targets = self._compute(outputs=outputs, targets=targets)

        true_preds = torch.eq(targets, outputs)
        false_preds = ~true_preds

        pos_preds = torch.eq(outputs, 1.0)
        neg_preds = torch.eq(outputs, 0.0)

        # Some einops operations
        pattern = "r c -> c" if self.average == "macro" else "r c -> "

        tp, fp, tn, fn = self._calculate_stats(true_preds, false_preds, pos_preds, neg_preds)

        # einops reductions
        tp = reduce(tp, pattern, reduction="sum")
        fp = reduce(fp, pattern, reduction="sum")
        tn = reduce(tn, pattern, reduction="sum")
        fn = reduce(fn, pattern, reduction="sum")

        return tp, fp, tn, fn

    def reduce(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Method to perform macro or micro reduction."""
        frac = numerator / (denominator + self.eps)
        return torch.mean(frac) if self.average == "macro" else frac

    def _compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self._check_type(outputs=outputs, targets=targets)

        if self.case_type == "multiclass":

            targets = self._convert_to_onehot(num_classes=self.num_classes, indices=targets.view(-1))
            # We receive logits  need argmax on preds
            outputs = torch.argmax(outputs, dim=1)
            outputs = self._convert_to_onehot(num_classes=self.num_classes, indices=outputs.view(-1))

        else:

            # Handling multilabel and binary cases

            outputs = torch.sigmoid(outputs).float()

            outputs = (outputs >= self.threshold).long()

        outputs = outputs.reshape(outputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        return outputs, targets


def calculate_segmentation_statistics(outputs: torch.Tensor, targets: torch.Tensor, class_dim: int = 1, threshold=None):
    """Compute calculate segmentation statistics.

    Args:
        outputs(torch.Tensor): torch.Tensor.
        targets(torch.Tensor): torch.Tensor.
        threshold(float): threshold for binarization of predictions.
        class_dim(int): indicates class dimension (K).

    Returns:
        True positives , false positives , false negatives for segmentation task.
    """
    num_dims = len(outputs.shape)

    assert num_dims > 2, "Found only two dimensions, shape should be [bs , C , ...]"  # noqa: S101

    assert outputs.shape == targets.shape, "shape mismatch"  # noqa: S101

    if threshold is not None:
        outputs = (outputs > threshold).float()

    dims = [dim for dim in range(num_dims) if dim != class_dim]

    true_positives = torch.sum(outputs * targets, dim=dims)
    false_positives = torch.sum(outputs * (1 - targets), dim=dims)
    false_negatives = torch.sum(targets * (1 - outputs), dim=dims)

    return true_positives, false_positives, false_negatives


class MetricMeter:
    """Base Class to structuring your metrics."""

    def accumulate(self, outputs, targets):
        """Method to accumulate outputs and targets per the batch."""
        raise NotImplementedError

    def reset(self):
        """Method to reset the accumulation lists."""
        raise NotImplementedError


__all__ = [
    "_BaseMetric",
    "_BaseInputHandler",
    "MetricMeter",
    "calculate_segmentation_statistics",
]

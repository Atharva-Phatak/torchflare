"""Implements variants for Cross Entropy loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def BCEWithLogitsFlat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Same as F.binary_cross_entropy_with_logits but flattens the input and target.

    Args:
        x : logits
        y: The corresponding targets.

    Returns:
        The computed Loss
    """
    y = y.view(x.shape).type_as(x)
    return torch.nn.functional.binary_cross_entropy_with_logits(x, y)


def BCEFlat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Same as F.binary_cross_entropy but flattens the input and target.

    Args:
        x : logits
        y: The corresponding targets.

    Returns:
        The computed Loss
    """
    x = torch.sigmoid(x)
    y = y.view(x.shape).type_as(x)
    return torch.nn.functional.binary_cross_entropy(x, y)


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with targets smoothing.

    Args:
           smoothing : targets smoothing factor

       Raises:
           ValueError: value error is raised if smoothing  > 1.0.
    """

    def __init__(self, smoothing: float = 0.1):
        """Constructor method for LabelSmoothingCrossEntropy."""
        super(LabelSmoothingCrossEntropy, self).__init__()
        if smoothing > 1.0:
            raise ValueError("Smoothing value must be less than 1.")
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            logits: Raw logits from the net.
            target: The targets.

        Returns:
            The computed loss value.
        """
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SymmetricCE(nn.Module):
    """Pytorch Implementation of Symmetric Cross Entropy.

    Paper: https://arxiv.org/abs/1908.06112

    Args:
            alpha: The alpha value for symmetricCE.
            beta: The beta value for symmetricCE.
            num_classes: The number of classes.
    """

    def __init__(self, num_classes, alpha: float = 1.0, beta: float = 1.0):
        """Constructor method for symmetric CE."""
        super(SymmetricCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        ce = self.ce(logits, targets)

        logits = F.softmax(logits, dim=1)
        logits = torch.clamp(logits, min=1e-7, max=1.0)
        if logits.is_cuda:
            label_one_hot = torch.nn.functional.one_hot(targets, self.num_classes).float().cuda()
        else:
            label_one_hot = torch.nn.functional.one_hot(targets, self.num_classes)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(logits * torch.log(label_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


__all__ = ["LabelSmoothingCrossEntropy", "SymmetricCE"]

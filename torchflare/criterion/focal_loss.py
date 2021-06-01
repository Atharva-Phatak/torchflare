"""Implements variants for Focal loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    """Implementation of Focal Loss for Binary Classification Problems.

    Focal loss was proposed in `Focal Loss for Dense Object Detection_.
    <https://arxiv.org/abs/1708.02002>`_.
    """

    def __init__(self, gamma=0, eps=1e-7, reduction="mean"):
        """Constructor Method for FocalLoss class.

        Args:
            gamma : The focal parameter. Defaults to 0.
            eps : Constant for computational stability.
            reduction: The reduction parameter for Cross Entropy Loss.
        """
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            logits: The raw logits from the network of shape (N,k) where C = number of classes , k = extra dims
            targets: The targets

        Returns:
            The computed loss value
        """
        targets = targets.view(logits.shape)
        logp = self.bce(logits, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


class FocalLoss(nn.Module):
    """Implementation of Focal Loss.

    Focal loss was proposed in `Focal Loss for Dense Object Detection_.
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
            gamma : The focal parameter. Defaults to 0.
            eps : Constant for computational stability.
            reduction: The reduction parameter for Cross Entropy Loss.
    """

    def __init__(self, gamma=0, eps=1e-7, reduction="mean"):
        """Constructor Method for FocalLoss class."""
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            logits: The raw logits from the network of shape (N,C,k) where C = number of classes and (k) = extra dims
            targets: The targets of shape (N , k).

        Returns:
            The computed loss value
        """
        logp = self.ce(logits, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss.

    Source : https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: float = 1, gamma: float = 2, xent: float = 0.1, reduction="mean"):
        """Constructor for FocalCosineLoss."""
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward Method."""
        cosine_loss = F.cosine_embedding_loss(
            logits,
            torch.nn.functional.one_hot(target, num_classes=logits.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(logits), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

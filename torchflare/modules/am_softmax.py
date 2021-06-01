"""Implements AM-softmax."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    """Implementation of `Additive Margin Softmax <https://arxiv.org/abs/1801.05599>`_.

    Args:
            in_features: Size of the input features
            out_features: The size of output features(usually number of num_classes)
            s: The norm for input features.
            m: margin

    """

    def __init__(self, in_features, out_features, m=0.35, s=32):
        """Class Constructor."""
        super(AMSoftmax, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s

        self.eps = 1e-7
        self.Weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.Weight)

    def forward(self, features: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Forward Pass.

        Args:
            features: The input features of shape (BS x F) where BS is batch size and F is input feature dimension.
            targets: The targets with shape BS , where BS is batch size

        Returns:
            Logits with shape (BS x out_features)
        """
        cos_theta = F.linear(F.normalize(features), F.normalize(self.Weight))
        if targets is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        logits = torch.where(one_hot.bool(), cos_theta - self.m, cos_theta)
        logits = torch.cos(logits)
        logits *= self.s

        return logits

"""Implements CosFace."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Implementation of `CosFace Loss  <https://arxiv.org/abs/1801.09414>`_.

    Args:
           in_features: Size of the input features
           out_features: The size of output features(usually number of num_classes)
           s: The norm for input features.
           m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        """Class Constructor."""
        super(CosFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
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
        # normalize features and weights
        logits = F.linear(F.normalize(features), F.normalize(self.Weight))
        if targets is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


__all__ = ["CosFace"]

"""Implements LiArcFace."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiArcFace(nn.Module):
    """Implementation of `Li-ArcFace <https://arxiv.org/abs/1907.12256>`_.

    Args:
           in_features: Size of the input features
           out_features: The size of output features(usually number of num_classes)
           s: The norm for input features.
           m: margin
    """

    def __init__(self, in_features, out_features, s=64, m=0.45):
        """Constructor class of LiArcFace."""
        super(LiArcFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
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

        cos_theta.clamp(-1 + self.eps, 1 - self.eps)

        theta = torch.acos(cos_theta)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.data.view(-1, 1), 1)
        target = (math.pi - 2 * (theta + self.m)) / math.pi
        other = (math.pi - 2 * theta) / math.pi

        output = (one_hot * target) + ((1.0 - one_hot) * other)

        output = output * self.s
        return output


__all__ = ["LiArcFace"]

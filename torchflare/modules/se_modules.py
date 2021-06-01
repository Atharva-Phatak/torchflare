"""Implementation of Squeeze and Excitation BLocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSE(nn.Module):
    """Implementation of Channel Wise Squeeze and Excitation Block.

    Paper : https://arxiv.org/abs/1709.01507

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    """

    def __init__(self, in_channels: int, r: int = 16):
        """Constructor for CSE class.

        Args:
            in_channels(int): The number of input channels in the feature map.
            r(int): The reduction ration (Default : 16)
        """
        super(CSE, self).__init__()

        self.in_channels = in_channels
        self.r = r
        self.linear1 = nn.Linear(self.in_channels, self.in_channels // self.r)
        self.linear2 = nn.Linear(self.in_channels // r, self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Method.

        Args:
            x(torch.Tensor): The input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of same shape
        """
        x_inp = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(x_inp, x)

        return x


class SSE(nn.Module):
    """SSE : Channel Squeeze and Spatial Excitation block.

    Paper : https://arxiv.org/abs/1803.02579

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    """

    def __init__(self, in_channels):
        """Constructor method for SSE class.

        Args:
            in_channels(int): The number of input channels in the feature map.
        """
        super(SSE, self).__init__()

        self.in_channels = in_channels
        # noinspection PyTypeChecker
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x) -> torch.Tensor:
        """Forward Method.

        Args:
            x(torch.Tensor): The input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of same shape
        """
        x_inp = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(x_inp, x)

        return x


class SCSE(nn.Module):
    """Implementation of SCSE : Concurrent Spatial and Channel Squeeze and Channel Excitation block.

    Paper : https://arxiv.org/abs/1803.02579


    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    """

    def __init__(self, in_channels, r=16):
        """Constructor for SCSE class.

        Args:
            in_channels(int): The number of input channels in the feature map.
            r(int): The reduction ration (Default : 16)
        """
        super(SCSE, self).__init__()

        self.in_channels = in_channels
        self.r = r

        self.cse = CSE(in_channels=self.in_channels, r=self.r)
        self.sse = SSE(in_channels=self.in_channels)

    def forward(self, x) -> torch.Tensor:
        """Forward method.

        Args:
            x(torch.Tensor): The input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of same shape
        """
        cse = self.cse(x)
        sse = self.sse(x)

        op = torch.add(cse, sse)

        return op


__all__ = ["SSE", "SCSE", "CSE"]

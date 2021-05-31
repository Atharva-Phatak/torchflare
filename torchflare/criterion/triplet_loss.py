"""Implements triplet loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchflare.criterion.utils import cosine_dist, euclidean_dist


def softmax_weights(dist, mask):

    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    difference = dist - max_v
    z = torch.sum(torch.exp(difference) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    weights = torch.exp(difference) * mask / z
    return weights


# Adapted From : https://github.com/earhian/Humpback-Whale-Identification-1st-/blob/master/models/triplet_loss.py
def hard_example_mining(distance_matrix, pos_idxs, neg_idxs):
    """For each anchor, find the hardest positive and negative sample.

    Args:
        distance_matrix: pair wise distance between samples, shape [N, M]
        pos_idxs: positive index with shape [N, M]
        neg_idxs: negative index with shape [N, M]

    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N];
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1

    Note:
        Only consider the case in which all targets have same num of samples,
        thus we can cope with all anchors in parallel.
    """
    assert len(distance_matrix.size()) == 2  # noqa: S101

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(distance_matrix * pos_idxs, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(distance_matrix * neg_idxs + pos_idxs * 99999999.0, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(distance_matrix, pos_idxs, neg_idxs):
    """For each anchor, find the weighted positive and negative sample.

    Args:
        distance_matrix: pytorch Variable, pair wise distance between samples, shape [N, N]
        pos_idxs:positive index with shape [N, M]
        neg_idxs: negative index with shape [N, M]

    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(distance_matrix.size()) == 2  # noqa: S101

    dist_ap = distance_matrix * pos_idxs
    dist_an = distance_matrix * neg_idxs

    weights_ap = softmax_weights(dist_ap, pos_idxs)
    weights_an = softmax_weights(-dist_an, neg_idxs)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Computes Triplet loss.

    Args:
            normalize_features: Whether to normalize the features. Default = True
            margin: The value for margin. Default = None.
            hard_mining: Whether to use hard sample mining. Default = True.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        margin: float = None,
        hard_mining: bool = True,
    ):
        """Constructor method for TripletLoss."""
        super(TripletLoss, self).__init__()

        self.normalize_features = normalize_features
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, embedding: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward Method.

        Args:
            embedding: The output of the network.
            targets: The targets.

        Returns:
            The computed Triplet Loss.
        """
        distance_matrix = (
            cosine_dist(embedding, embedding) if self.normalize_features else euclidean_dist(embedding, embedding)
        )

        n = distance_matrix.size(0)
        pos_idxs = targets.view(n, 1).expand(n, n).eq(targets.view(n, 1).expand(n, n).t()).float()
        neg_idxs = targets.view(n, 1).expand(n, n).ne(targets.view(n, 1).expand(n, n).t()).float()

        if self.hard_mining:

            dist_ap, dist_an = hard_example_mining(
                distance_matrix=distance_matrix, pos_idxs=pos_idxs, neg_idxs=neg_idxs
            )

        else:

            dist_ap, dist_an = weighted_example_mining(
                distance_matrix=distance_matrix, pos_idxs=pos_idxs, neg_idxs=neg_idxs
            )

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None and self.margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float("Inf"):
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss


__all__ = ["TripletLoss"]

"""Imports for criterion."""
from torchflare.criterion.cross_entropy import (
    BCEFlat,
    BCEWithLogitsFlat,
    LabelSmoothingCrossEntropy,
    SymmetricCE,
)
from torchflare.criterion.focal_loss import BCEFocalLoss, FocalCosineLoss, FocalLoss
from torchflare.criterion.triplet_loss import TripletLoss

__all__ = [
    "LabelSmoothingCrossEntropy",
    "SymmetricCE",
    "FocalLoss",
    "FocalCosineLoss",
    "TripletLoss",
    "BCEFocalLoss",
    "BCEWithLogitsFlat",
    "BCEFlat",
]

import torch
import torchflare.criterion as crit
import numpy as np
import pytest
import torch.nn.functional as F


def test_focal_loss():

    x = torch.randn(2, 4)
    y = torch.tensor([1, 2])
    ce = torch.nn.CrossEntropyLoss()(x, y)
    focal = crit.FocalLoss(gamma=0.0)(x, y)

    assert focal.item() == pytest.approx(ce.item(), 1e-4)


def test_bce_focal_loss():
    x = torch.randn(2, 1)
    y = torch.empty(2).random_(2)

    bce = torch.nn.BCEWithLogitsLoss()(x, y.view(x.shape))
    focal = crit.BCEFocalLoss(gamma=0.0)(x, y)

    assert focal.item() == pytest.approx(bce.item(), 1e-4)


def test_bce_variants():
    x = torch.randn(2, 1)
    y = torch.empty(2).random_(2)

    actual_bce_logits = F.binary_cross_entropy_with_logits(x, y.view(x.shape))
    bce_logits = crit.BCEWithLogitsFlat(x, y)

    assert bce_logits.item() == pytest.approx(actual_bce_logits.item(), 1e-4)

    actual_bce = F.binary_cross_entropy(torch.sigmoid(x), y.view(x.shape))
    bce = crit.BCEFlat(x, y)

    assert bce.item() == pytest.approx(actual_bce.item(), 1e-4)


def test_label_smoothing_ce():

    soft_ce_criterion = crit.LabelSmoothingCrossEntropy(smoothing=0.0)
    ce_criterion = torch.nn.CrossEntropyLoss()

    y_pred = torch.tensor([[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]]).float()
    y_true = torch.tensor([0, 1, 2, 3]).long()

    actual = soft_ce_criterion(y_pred, y_true).item()
    expected = ce_criterion(y_pred, y_true).item()
    np.testing.assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "criterion", [crit.TripletLoss(), crit.FocalCosineLoss()],
)
def test_forward_passes(criterion):

    output = torch.randn(3, 5)
    targets = torch.empty(3, dtype=torch.long).random_(5)

    loss = criterion(output, targets)
    assert torch.is_tensor(loss) == True

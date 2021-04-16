# flake8: noqa
import pytest
import torch

from torchflare.metrics import IOU, DiceScore

base_outputs = torch.tensor([[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]])
base_targets = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1, 1, 0]])
base_outputs = torch.stack([base_outputs, base_targets])[None, :, :, :]
base_targets = torch.stack([base_targets, base_targets])[None, :, :, :]
EPS = 1e-5


def test_dice():

    actual_dice = 0.6818181872367859
    dc = DiceScore()
    dc.reset()
    dc.accumulate(base_outputs, base_targets)

    assert actual_dice == pytest.approx(dc.compute().item())


def test_iou():

    actual_iou = 0.6111111044883728

    iou = IOU()
    iou.reset()
    iou.accumulate(base_outputs, base_targets)

    assert actual_iou == pytest.approx(iou.compute().item())

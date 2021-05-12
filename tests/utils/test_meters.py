import pytest

from torchflare.utils.average_meter import AverageMeter


def test_avg_meter():
    x = 0
    meter = AverageMeter()
    for i in range(100):
        x += i
        meter.update(i, 1)

    avg = x / 100
    assert avg == pytest.approx(meter.avg)

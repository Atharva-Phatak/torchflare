import pytest

from torchflare.utils.average_meter import AverageMeter
from torchflare.utils.imports_check import module_available

def test_avg_meter():
    x = 0
    count = 0
    meter = AverageMeter()
    for i in range(100):
        meter.update(i, 1)
        x += i
        count += 1
        assert meter.val == i

    avg = x / 100
    assert avg == pytest.approx(meter.avg)
    assert x == meter.sum
    assert count == meter.count

    meter.reset()
    assert meter.count == 0
    assert meter.avg == 0
    assert meter.val == 0
    assert meter.sum == 0


@pytest.mark.parametrize("modules" , ["torch" , "lightly"])
def test_imports(modules):

    available = module_available(modules)
    if modules == "torch":
        assert available is True
    else:
        assert available is False

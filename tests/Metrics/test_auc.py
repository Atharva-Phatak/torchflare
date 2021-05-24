from torchflare.metrics.auc import AUC
import torch
import numpy as np
import pytest
import sklearn.metrics as metrics

NUM_BATCHES = 10
BATCH_SIZE = 200


def _create_batches():
    for i in range(NUM_BATCHES):
        x = np.random.rand(NUM_BATCHES * BATCH_SIZE)
        y = np.random.rand(NUM_BATCHES * BATCH_SIZE)
        idx = np.argsort(x, kind="stable")
        x = x[idx] if i % 2 == 0 else x[idx[::-1]]
        y = y[idx] if i % 2 == 0 else x[idx[::-1]]
        x = x.reshape(NUM_BATCHES, BATCH_SIZE)
        y = y.reshape(NUM_BATCHES, BATCH_SIZE)

    return x, y


def sk_auc(x, y):
    x = x.flatten()
    y = y.flatten()

    return metrics.auc(x, y)


def test_base_auc():
    x, y = _create_batches()
    x, y = torch.tensor(x), torch.tensor(y)
    auc = AUC()
    auc.accumulate(x, y)
    assert auc.value.item() == pytest.approx(sk_auc(x, y))

@pytest.mark.parametrize(['x', 'y', 'expected'], [
        pytest.param([0, 1], [0, 1], 0.5),
        pytest.param([1, 0], [0, 1], 0.5),
        pytest.param([1, 0, 0], [0, 1, 1], 0.5),
        pytest.param([0, 1], [1, 1], 1),
        pytest.param([0, 0.5, 1], [0, 0.5, 1], 0.5),
    ])
def test_auc(x, y, expected):
    x = torch.tensor(x)
    y = torch.tensor(y)

    auc = AUC()
    auc.accumulate(x, y)
    assert auc.value.item() == expected

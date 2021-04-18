# flake8: noqa
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score

from torchflare.metrics.accuracy_meter import Accuracy

torch.manual_seed(42)
def test_binary_inputs():
    def _test(num_classes, threshold, multilabel):

        acc = Accuracy(num_classes=num_classes, threshold=threshold, multilabel=multilabel)

        acc.reset()
        outputs = torch.randn(100, 1)
        targets = torch.randint(0, 2, size=(100,))

        np_outputs = (torch.sigmoid(outputs) > threshold).float().numpy().flatten()
        np_targets = targets.numpy()

        acc.accumulate(outputs=outputs, targets=targets)

        acc_val = acc.compute()
        assert acc.case_type == "binary"
        # print(acc_val)
        acc_skm = accuracy_score(np_targets, np_outputs)
        assert acc_skm == pytest.approx(acc_val.item())

        bs = 16
        iters = targets.shape[0] // bs + 1

        acc.reset()
        for i in range(iters):
            idx = i * bs
            acc.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

        m_acc = acc.compute()
        assert acc_skm == pytest.approx(m_acc.item())

    for _ in range(10):
        _test(num_classes=2, threshold=0.5, multilabel=False)


def test_multiclass_inputs():
    def _test(num_classes, threshold, multilabel):

        acc = Accuracy(num_classes=num_classes, threshold=threshold, multilabel=multilabel)

        acc.reset()
        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 4, size=(100,))

        np_outputs = torch.argmax(outputs, dim=1).numpy()
        np_targets = targets.numpy()

        acc.accumulate(outputs=outputs, targets=targets)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            acc_skm = accuracy_score(np_targets, np_outputs)

        acc_val = acc.compute()
        assert acc.case_type == "multiclass"

        assert acc_skm == pytest.approx(acc_val.item())

        acc.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        acc.reset()
        for i in range(iters):
            idx = i * bs
            acc.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

        m_acc = acc.compute()
        assert acc_skm == pytest.approx(m_acc.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=False)


def test_multilabel_inputs():

    def _test(num_classes, threshold, multilabel):

        acc = Accuracy(num_classes=num_classes, threshold=threshold, multilabel=multilabel)

        acc.reset()
        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 2, size=(100, 4))

        np_outputs = (torch.sigmoid(outputs) >= threshold).int()
        np_targets = targets
        acc.accumulate(outputs=outputs, targets=targets)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            acc_skm = accuracy_score(np_targets, np_outputs)

        acc_val = acc.compute()
        assert acc.case_type == "multilabel"

        assert acc_skm == pytest.approx(acc_val.item())

        acc.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        acc.reset()
        for i in range(iters):
            idx = i * bs
            acc.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])
        m_acc = acc.compute()
        assert acc_skm == pytest.approx(m_acc.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=True)

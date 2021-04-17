# flake8: noqa
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import recall_score

from torchflare.metrics.recall_meter import Recall
from torchflare.metrics.meters import _BaseInputHandler

torch.manual_seed(42)


def test_binary_inputs():
    def _test(num_classes, threshold, multilabel, average):

        rc = Recall(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)
        outputs = torch.randn(100, 1)
        targets = torch.randint(0, 2, size=(100,))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)
        rc.accumulate(outputs=outputs, targets=targets)
        rec_val = rc.compute()
        assert rc.case_type == "binary"

        rec_skm = recall_score(np_targets.numpy(), np_outputs.numpy(), average="binary")

        assert rec_skm == pytest.approx(rec_val.item())

        rc.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            rc.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

        m_rc = rc.compute()
        assert rec_skm == pytest.approx(m_rc.item())

    for _ in range(10):
        _test(num_classes=1, threshold=0.5, multilabel=False, average="macro")
        _test(num_classes=1, threshold=0.5, multilabel=False, average="micro")


def test_multiclass_inputs():
    def _test(num_classes, threshold, multilabel, average):

        rc = Recall(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 4, size=(100,))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        rc.accumulate(outputs=outputs, targets=targets)
        rec_val = rc.compute()
        assert rc.case_type == "multiclass"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            rec_skm = recall_score(np_targets.numpy(), np_outputs.numpy(), average=average)

        assert rec_skm == pytest.approx(rec_val.item())

        rc.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            rc.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

        rec_m = rc.compute()
        assert rec_skm == pytest.approx(rec_m.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=False, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=False, average="micro")


def test_multilabel_inputs():
    def _test(num_classes, threshold, multilabel, average):

        rc = Recall(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 2, size=(100, 4))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        rc.accumulate(outputs=outputs, targets=targets)
        rec_val = rc.compute()

        assert rc.case_type == "multilabel"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            rec_skm = recall_score(np_targets, np_outputs, average=average)

        assert rec_skm == pytest.approx(rec_val.item())
        rc.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            rc.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )

        m_rc = rc.compute()
        assert rec_skm == pytest.approx(m_rc.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=True, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=True, average="micro")

# flake8: noqa
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score

from torchflare.metrics.precision_meter import Precision
from torchflare.metrics.meters import _BaseInputHandler

torch.manual_seed(42)


def test_binary_inputs():
    def _test(out_features, threshold, multilabel, average):

        pr = Precision(num_classes=out_features, threshold=threshold, multilabel=multilabel, average=average,)
        outputs = torch.randn(10, 1)
        targets = torch.randint(0, 2, size=(10,))

        pr.reset()
        bs = _BaseInputHandler(num_classes=out_features, threshold=0.5, multilabel=multilabel, average=average,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        pr.accumulate(outputs=outputs, targets=targets)
        pr_val = pr.compute()
        assert pr.case_type == "binary"

        pr_skm = precision_score(np_targets.numpy(), np_outputs.numpy(), average="binary")
        assert pr_skm == pytest.approx(pr_val.item())

        pr.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            pr.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )

        m_pr = pr.compute()
        assert pr_skm == pytest.approx(m_pr.item())

    for _ in range(10):
        _test(out_features=2, threshold=0.5, multilabel=False, average="macro")
        _test(out_features=2, threshold=0.5, multilabel=False, average="micro")


def test_multiclass_inputs():
    def _test(num_classes, threshold, multilabel, average):

        pr = Precision(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 4, size=(100,))

        bs = _BaseInputHandler(num_classes=num_classes, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        pr.accumulate(outputs=outputs, targets=targets)

        pr_val = pr.compute()

        assert pr.case_type == "multiclass"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            pr_skm = precision_score(np_targets.numpy(), np_outputs.numpy(), average=average)

        assert pr_skm == pytest.approx(pr_val.item())

        pr.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            pr.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )
        pr_meter_val = pr.compute()
        assert pr_skm == pytest.approx(pr_meter_val.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=False, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=False, average="micro")


def test_multilabel_inputs():
    def _test(num_classes, threshold, multilabel, average):

        pr = Precision(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 2, size=(100, 4))

        bs = _BaseInputHandler(num_classes=num_classes, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        pr.accumulate(outputs=outputs, targets=targets)

        pr_val = pr.compute()

        assert pr.case_type == "multilabel"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)

            pr_skm = precision_score(np_targets.numpy(), np_outputs.numpy(), average=average)

        assert pr_skm == pytest.approx(pr_val.item())

        pr.reset()
        bs = 16
        iters = targets.shape[0] // bs + 1

        for i in range(iters):
            idx = i * bs
            pr.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )
        pr_meter_val = pr.compute()
        assert pr_skm == pytest.approx(pr_meter_val.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=True, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=True, average="micro")

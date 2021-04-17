# flake8: noqa
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, fbeta_score

from torchflare.metrics.fbeta_meter import F1Score, FBeta, _BaseInputHandler

torch.manual_seed(42)


def test_binary_inputs():
    def _test(num_classes, threshold, multilabel, average):

        fbeta = FBeta(beta=2.0, num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        f1 = F1Score(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 1)
        targets = torch.randint(0, 2, size=(100,))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        fbeta.accumulate(outputs=outputs, targets=targets)
        f1.accumulate(outputs=outputs, targets=targets)

        fbeta_val = fbeta.compute()
        f1_val = f1.compute()

        assert fbeta.case_type == "binary"
        assert f1.case_type == "binary"

        fbeta_skm = fbeta_score(np_targets.numpy(), np_outputs.numpy(), average="binary", beta=2.0)

        f1_skm = f1_score(np_targets.numpy(), np_outputs.numpy(), average="binary")

        assert fbeta_skm == pytest.approx(fbeta_val.item())
        assert f1_skm == pytest.approx(f1_val.item())

        bs = 16
        iters = targets.shape[0] // bs + 1

        fbeta.reset()
        f1.reset()
        for i in range(iters):
            idx = i * bs

            fbeta.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

            f1.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )

        f1_m = f1.compute()
        fbeta_m = fbeta.compute()

        assert f1_skm == pytest.approx(f1_m.item())

        assert fbeta_skm == pytest.approx(fbeta_m.item())

    for _ in range(10):
        _test(num_classes=1, threshold=0.5, multilabel=False, average="macro")
        _test(num_classes=1, threshold=0.5, multilabel=False, average="micro")


def test_multiclass_inputs():
    def _test(num_classes, threshold, multilabel, average):

        fbeta = FBeta(beta=2.0, num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        f1 = F1Score(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(100, 4)
        targets = torch.randint(0, 4, size=(100,))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        fbeta.accumulate(outputs=outputs, targets=targets)
        f1.accumulate(outputs=outputs, targets=targets)

        fbeta_val = fbeta.compute()
        f1_val = f1.compute()

        assert fbeta.case_type == "multiclass"
        assert f1.case_type == "multiclass"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            fbeta_skm = fbeta_score(np_targets.numpy(), np_outputs.numpy(), average=average, beta=2.0)

            f1_skm = f1_score(np_targets.numpy(), np_outputs.numpy(), average=average)

        assert fbeta_skm == pytest.approx(fbeta_val.item())
        assert f1_skm == pytest.approx(f1_val.item())

        bs = 16
        iters = targets.shape[0] // bs + 1

        fbeta.reset()
        f1.reset()
        for i in range(iters):
            idx = i * bs

            fbeta.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

            f1.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )

        f1_m = f1.compute()
        fbeta_m = fbeta.compute()

        assert f1_skm == pytest.approx(f1_m.item())

        assert fbeta_skm == pytest.approx(fbeta_m.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=False, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=False, average="micro")


def test_multilabel_inputs():
    def _test(num_classes, threshold, multilabel, average):

        fbeta = FBeta(beta=2.0, num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        f1 = F1Score(num_classes=num_classes, threshold=threshold, multilabel=multilabel, average=average,)

        outputs = torch.randn(10, 4)
        targets = torch.randint(0, 2, size=(10, 4))

        bs = _BaseInputHandler(num_classes=num_classes, average=average, threshold=0.5, multilabel=multilabel,)

        np_outputs, np_targets = bs._compute(outputs=outputs, targets=targets)

        fbeta.accumulate(outputs=outputs, targets=targets)
        f1.accumulate(outputs=outputs, targets=targets)

        fbeta_val = fbeta.compute()
        f1_val = f1.compute()

        assert fbeta.case_type == "multilabel"
        assert f1.case_type == "multilabel"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            fbeta_skm = fbeta_score(np_targets.numpy(), np_outputs.numpy(), average=average, beta=2.0)

            f1_skm = f1_score(np_targets.numpy(), np_outputs.numpy(), average=average)

        assert fbeta_skm == pytest.approx(fbeta_val.item())
        assert f1_skm == pytest.approx(f1_val.item())

        bs = 16
        iters = targets.shape[0] // bs + 1

        fbeta.reset()
        f1.reset()
        for i in range(iters):
            idx = i * bs
            fbeta.accumulate(outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs])

            f1.accumulate(
                outputs=outputs[idx : idx + bs], targets=targets[idx : idx + bs],
            )

        f1_m = f1.compute()
        fbeta_m = fbeta.compute()

        assert f1_skm == pytest.approx(f1_m.item())

        assert fbeta_skm == pytest.approx(fbeta_m.item())

    for _ in range(10):
        _test(num_classes=4, threshold=0.5, multilabel=True, average="macro")
        _test(num_classes=4, threshold=0.5, multilabel=True, average="micro")

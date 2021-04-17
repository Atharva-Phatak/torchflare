# flake8: noqa
import collections

import pytest
import sklearn.metrics as skm
import torch

from torchflare.metrics.regression import MAE, MSE, MSLE, R2Score

torch.manual_seed(42)

n_targets = 3

inputs = collections.namedtuple("input", ["outputs", "targets"])
single_target_inputs = inputs(outputs=torch.rand(10, 4), targets=torch.rand(10, 4))

multi_target_inputs = inputs(outputs=torch.rand(10, 4, n_targets), targets=torch.rand(10, 4, n_targets))


def test_mse():
    def _test_single_target():
        np_outputs = single_target_inputs.outputs.view(-1).numpy()
        np_targets = single_target_inputs.targets.view(-1).numpy()

        mse = MSE()

        mse.accumulate(outputs=single_target_inputs.outputs, targets=single_target_inputs.targets)

        assert skm.mean_squared_error(np_targets, np_outputs) == pytest.approx(mse.compute().item())

    def _test_multiple_target():
        np_outputs = multi_target_inputs.outputs.view(-1, n_targets).numpy()
        np_targets = multi_target_inputs.targets.view(-1, n_targets).numpy()

        mse = MSE()

        mse.accumulate(outputs=multi_target_inputs.outputs, targets=multi_target_inputs.targets)

        assert skm.mean_squared_error(np_targets, np_outputs) == pytest.approx(mse.compute().item())

    for _ in range(10):
        _test_single_target()
        _test_multiple_target()


def test_mae():
    def _test_single_target():
        np_outputs = single_target_inputs.outputs.view(-1).numpy()
        np_targets = single_target_inputs.targets.view(-1).numpy()

        mae = MAE()

        mae.accumulate(outputs=single_target_inputs.outputs, targets=single_target_inputs.targets)

        assert skm.mean_absolute_error(np_targets, np_outputs) == pytest.approx(mae.compute().item())

    def _test_multiple_target():
        np_outputs = multi_target_inputs.outputs.view(-1, n_targets).numpy()
        np_targets = multi_target_inputs.targets.view(-1, n_targets).numpy()

        mae = MAE()

        mae.accumulate(outputs=multi_target_inputs.outputs, targets=multi_target_inputs.targets)

        assert skm.mean_absolute_error(np_targets, np_outputs) == pytest.approx(mae.compute().item())

    for _ in range(10):
        _test_single_target()
        _test_multiple_target()


def test_msle():
    def _test_single_target():
        np_outputs = single_target_inputs.outputs.view(-1).numpy()
        np_targets = single_target_inputs.targets.view(-1).numpy()

        msle = MSLE()

        msle.accumulate(outputs=single_target_inputs.outputs, targets=single_target_inputs.targets)

        assert skm.mean_squared_log_error(np_targets, np_outputs) == pytest.approx(msle.compute().item())

    def _test_multiple_target():
        np_outputs = multi_target_inputs.outputs.view(-1, n_targets).numpy()
        np_targets = multi_target_inputs.targets.view(-1, n_targets).numpy()

        msle = MSLE()

        msle.accumulate(outputs=multi_target_inputs.outputs, targets=multi_target_inputs.targets)

        assert skm.mean_squared_log_error(np_targets, np_outputs) == pytest.approx(msle.compute().item())

    for _ in range(10):
        _test_single_target()
        _test_multiple_target()


def test_r2score():
    def _test():

        size = 51
        preds = torch.rand(size)
        targets = torch.rand(size)
        np_y_pred = preds.numpy()
        np_y = targets.numpy()

        m = R2Score()

        m.reset()
        m.accumulate(preds, targets)
        assert skm.r2_score(np_y, np_y_pred) == pytest.approx(m.compute().item(), abs=1e-4)

    for _ in range(10):
        _test()

# noqa
from collections import Counter

import pytest
import torch
import torch.optim as optim
import torch.optim.optimizer as Optimizer
from torch.optim import lr_scheduler

from torchflare.callbacks.lr_schedulers import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LRSchedulerCallback,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class TestExp:
    def __init__(self, cb):

        self.model = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03)
        self.exp_logs = None
        self.main_metric = "loss"
        self.cb = cb
        self.cb.set_experiment(self)
        self.val_key = "val_"

    def run(self):

        self.cb.experiment_start()
        loss = 0.1
        for _ in range(5):
            for i in range(2):
                loss = loss * 2
                self.cb.batch_end()

            self.exp_logs = {"val_loss": loss}
            self.cb.epoch_end()


class MockScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        self.step_count += 1


@pytest.fixture(scope="function", params=[True, False])
def step_on_batch(request):
    return request.param


def test_lr_scheduler():
    scheduler = LRSchedulerCallback(MockScheduler, step_on_batch=False)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert dummy.cb.scheduler.step_count == 5
    assert isinstance(dummy.cb.scheduler.optimizer, torch.optim.Optimizer) is True


def test_lr_scheduler_step():
    scheduler = LRSchedulerCallback(MockScheduler, step_on_batch=True)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert dummy.cb.scheduler.step_count == 10
    assert isinstance(dummy.cb.scheduler.optimizer, torch.optim.Optimizer) is True


def test_lambda_lr():

    scheduler = LambdaLR(lr_lambda=lambda epoch: 0.95 ** epoch, step_on_batch=step_on_batch)

    dummy = TestExp(cb=scheduler)
    dummy.run()

    assert isinstance(scheduler.scheduler, lr_scheduler._LRScheduler) is True
    assert scheduler.scheduler.lr_lambdas[0](1) == 0.95 ** 1
    assert scheduler.step_on_batch == step_on_batch


def test_step_lr():

    scheduler = StepLR(step_size=10, gamma=0.1)

    dummy = TestExp(cb=scheduler)
    dummy.run()

    assert isinstance(scheduler.scheduler, lr_scheduler.StepLR) is True
    assert scheduler.scheduler.step_size == 10
    assert scheduler.scheduler.gamma == 0.1


def test_multistep_lr():

    scheduler = MultiStepLR(milestones=[30, 80], gamma=0.1, step_on_batch=step_on_batch)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.MultiStepLR) is True
    assert scheduler.scheduler.milestones == Counter([30, 80])
    assert scheduler.scheduler.gamma == 0.1
    assert scheduler.step_on_batch == step_on_batch


def test_exponential_lr():
    scheduler = ExponentialLR(gamma=0.1, step_on_batch=step_on_batch)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.ExponentialLR)
    assert scheduler.scheduler.gamma == 0.1
    assert scheduler.step_on_batch == step_on_batch


def test_cosine_annealing_lr():
    scheduler = CosineAnnealingLR(T_max=10, eta_min=0)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.CosineAnnealingLR)
    assert scheduler.scheduler.T_max == 10
    assert scheduler.scheduler.eta_min == 0


def test_multipilicative_lr():
    scheduler = MultiplicativeLR(lambda epoch: 0.95, step_on_batch=step_on_batch)
    dummy = TestExp(cb=scheduler)
    dummy.run()

    assert isinstance(scheduler.scheduler, lr_scheduler.MultiplicativeLR)
    assert scheduler.scheduler.lr_lambdas[0](1) == 0.95
    assert scheduler.step_on_batch == step_on_batch


def test_one_cycle():
    scheduler = OneCycleLR(max_lr=0.01, steps_per_epoch=1000, epochs=10)
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.OneCycleLR)
    assert scheduler.scheduler.total_steps == 10000


def test_cosine_annealing_warm_restarts():
    scheduler = CosineAnnealingWarmRestarts(T_0=1, T_mult=1, eta_min=0, step_on_batch=step_on_batch)
    dummy = TestExp(cb=scheduler)
    dummy.run()

    assert isinstance(scheduler.scheduler, lr_scheduler.CosineAnnealingWarmRestarts)
    assert scheduler.scheduler.T_0 == 1
    assert scheduler.scheduler.T_mult == 1
    assert scheduler.scheduler.eta_min == 0
    assert scheduler.step_on_batch == step_on_batch


def test_cyclic_lr():
    scheduler = CyclicLR(
        base_lr=0.001,
        max_lr=0.01,
        gamma=1.0,
        mode="triangular",
        scale_mode="cycle",
        cycle_momentum=True,
        step_on_batch=step_on_batch,
    )
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.CyclicLR)
    assert scheduler.scheduler.base_lrs == [0.001]
    assert scheduler.scheduler.max_lrs == [0.01]
    assert scheduler.scheduler.gamma == 1.0
    assert scheduler.scheduler.mode == "triangular"
    assert scheduler.scheduler.scale_mode == "cycle"
    assert scheduler.scheduler.cycle_momentum
    assert scheduler.step_on_batch == step_on_batch


def test_reduce_lr_on_plateau():
    scheduler = ReduceLROnPlateau(
        mode="min", factor=0.1, patience=3, threshold=1e-6, threshold_mode="rel", cooldown=0, eps=1e-8
    )
    dummy = TestExp(cb=scheduler)
    dummy.run()
    assert isinstance(scheduler.scheduler, lr_scheduler.ReduceLROnPlateau)
    assert scheduler.scheduler.mode == "min"
    assert scheduler.scheduler.factor == 0.1
    assert scheduler.scheduler.patience == 3
    assert scheduler.scheduler.threshold == 1e-6
    assert scheduler.scheduler.threshold_mode == "rel"
    assert scheduler.scheduler.cooldown == 0
    assert scheduler.scheduler.eps == 1e-8
    assert not scheduler.step_on_batch

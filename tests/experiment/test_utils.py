# flake8: noqa
from torchflare.experiments.criterion_utilities import get_criterion
from torchflare.experiments.optim_utilities import get_optimizer
from torchflare.experiments.scheduler_utilities import get_scheduler
import pytest

def test_utils():
    def get_val_str():
        opt = "SGD"
        sch = "get_constant_schedule_with_warmup"
        loss = "binary_cross_entropy"

        loss_fn = get_criterion(loss)
        optim = get_optimizer(opt)
        scheduler = get_scheduler(sch)
        assert loss_fn.__name__ == loss
        assert optim.__name__ == opt
        assert scheduler.__name__ == sch

        with pytest.raises(ValueError):
            invalid_sch = "mstep_lr"
            invalid_optim = "adm"
            invalid_crit = "bcelogits"
            scheduler = get_scheduler(invalid_sch)
            optim = get_optimizer(invalid_optim)
            scheduler = get_criterion(invalid_crit)

    get_val_str()

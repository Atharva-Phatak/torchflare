from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Dict, Union

import torch.nn as nn
from torch.optim import Optimizer

from torchflare.experiments.simple_utils import check_both_dicts, check_same_keys


def _validate_inputs(d1, d2):
    if check_both_dicts(d1, d2):
        return check_same_keys(d1, d2)
    return False


@dataclass
class ModelConfig:
    """Model Config to initialize model related parameters, optimizers and criterion."""

    nn_module: Union[nn.Module, Dict] = field(
        default=MISSING,
        metadata={
            "help": "An uninstantiated PyTorch class or a dictionary of uninstantiated PyTorch classes \
            which defines the model."
        },
    )

    module_params: Dict = field(default=MISSING, metadata={"help": "The params required to initialize model class."})

    optimizer: Union[str, Dict, Optimizer, Any] = field(
        default=MISSING,
        metadata={
            "help": "The optimizer class to be used or name of optimizer or dict of optimizers \
                        If you pass in the name of the optimizer, only optimizers available in pytorch are supported."
        },
    )

    optimizer_params: Dict = field(default=MISSING, metadata={"help": "The parameters to instantiate optimizer."})

    criterion: Union[Callable, Dict, str] = field(
        default=MISSING,
        metadata={
            "help": "The loss function to optimize or name of the loss function.\
                    If you pass in the name of the loss function,\
                    only loss functions available in pytorch can be supported."
        },
    )
    model_dict: bool = field(default=False)
    optimizer_dict: bool = field(default=False)

    def __post_init__(self):
        """Post initialisation checks."""
        self.model_dict = _validate_inputs(d1=self.nn_module, d2=self.module_params)
        self.optimizer_dict = _validate_inputs(d1=self.optimizer, d2=self.optimizer_params)

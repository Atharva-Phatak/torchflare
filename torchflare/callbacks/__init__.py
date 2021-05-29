# flake8: noqa
# isort: skip
"""Imports for callbacks."""
from torchflare.callbacks.callback import Callbacks, sort_callbacks  # isort: skip
from torchflare.callbacks.comet_logger import CometLogger  # isort: skip
from torchflare.callbacks.early_stopping import EarlyStopping  # isort: skip
from torchflare.callbacks.extra_utils import init_improvement  # isort: skip
from torchflare.callbacks.load_checkpoint import LoadCheckpoint  # isort: skip
from torchflare.callbacks.lr_schedulers import CosineAnnealingWarmRestarts  # isort: skip
from torchflare.callbacks.lr_schedulers import CyclicLR  # isort: skip
from torchflare.callbacks.lr_schedulers import ExponentialLR  # isort: skip
from torchflare.callbacks.lr_schedulers import LambdaLR  # isort: skip
from torchflare.callbacks.lr_schedulers import LRSchedulerCallback  # isort: skip
from torchflare.callbacks.lr_schedulers import MultiplicativeLR  # isort: skip
from torchflare.callbacks.lr_schedulers import MultiStepLR  # isort: skip
from torchflare.callbacks.lr_schedulers import OneCycleLR  # isort: skip
from torchflare.callbacks.lr_schedulers import ReduceLROnPlateau  # isort: skip
from torchflare.callbacks.lr_schedulers import StepLR  # isort: skip
from torchflare.callbacks.message_notifiers import DiscordNotifierCallback, SlackNotifierCallback
from torchflare.callbacks.metric_utils import MetricCallback
from torchflare.callbacks.model_checkpoint import ModelCheckpoint  # isort: skip
from torchflare.callbacks.model_history import History  # isort: skip
from torchflare.callbacks.neptune_logger import NeptuneLogger
from torchflare.callbacks.progress_bar import ProgressBar
from torchflare.callbacks.states import CallbackOrder  # isort: skip
from torchflare.callbacks.tensorboard_logger import TensorboardLogger
from torchflare.callbacks.wandb_logger import WandbLogger  # isort: skip

from torchflare.callbacks.lr_schedulers import CosineAnnealingLR  # isort: skip; isort: skip


__all__ = [
    "Callbacks",
    "CallbackOrder",
    "sort_callbacks",
    "EarlyStopping",
    "SlackNotifierCallback",
    "DiscordNotifierCallback",
    "CometLogger",
    "NeptuneLogger",
    "init_improvement",
    "TensorboardLogger",
    "WandbLogger",
    "MetricCallback",
    "ModelCheckpoint",
    "History",
    "LoadCheckpoint",
    "LRSchedulerCallback",
    "LambdaLR",
    "OneCycleLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "MultiplicativeLR",
    "MultiStepLR",
    "ReduceLROnPlateau",
    "StepLR",
    "ExponentialLR",
    "ProgressBar",
]

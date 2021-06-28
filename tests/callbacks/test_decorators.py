from torchflare.callbacks.callback_decorators import (
    bind_to_event,
    on_batch_end,
    on_epoch_start,
    on_epoch_end,
    on_loader_end,
    on_loader_start,
    on_batch_start,
    on_experiment_start,
    on_experiment_end,
)
from torchflare.callbacks.states import CallbackOrder
from torchflare.callbacks.callback import Callbacks

order = CallbackOrder.MODEL_INIT
state = "test"


def example(state):
    return state


def test_bind_to_event():
    assert bind_to_event(Callbacks.on_experiment_start)(example, order).on_experiment_start(state) == state
    assert bind_to_event(Callbacks.on_experiment_end)(example, order).on_experiment_end(state) == state


def test_decorators():
    assert on_experiment_start(order)(example).on_experiment_start(state) == state
    assert on_experiment_end(order)(example).on_experiment_end(state) == state
    assert on_epoch_start(order)(example).on_epoch_start(state) == state
    assert on_epoch_end(order)(example).on_epoch_end(state) == state
    assert on_loader_start(order)(example).on_loader_start(state) == state
    assert on_loader_end(order)(example).on_loader_end(state) == state
    assert on_batch_start(order)(example).on_batch_start(state) == state
    assert on_batch_end(order)(example).on_batch_end(state) == state

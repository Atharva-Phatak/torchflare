import functools
import types
from typing import TYPE_CHECKING

from torchflare.callbacks.callback import Callbacks
from torchflare.callbacks.states import CallbackOrder

if TYPE_CHECKING:
    from torchflare.experiments.experiment import Experiment


class FunctionalCallback(Callbacks):
    """Utility class to call the input function.

    Args:
        func: The input function.
        order: The CallbackOrder.
    """

    def __init__(self, func, order):
        super(FunctionalCallback, self).__init__(order=order)
        self.func = func
        functools.update_wrapper(self, func)

    def on_event(self, experiment: "Experiment"):
        """Call the function."""
        return self.func(experiment)


def bind_to_event(event):
    """Method to bind functions to event."""

    @functools.wraps(event)
    def decorator(func, order):
        callback = FunctionalCallback(func=func, order=order)
        setattr(
            callback,
            event.__name__,
            types.MethodType(
                lambda self, experiment: self.on_event(experiment=experiment), callback
            ),
        )
        return callback

    return decorator


def on_experiment_start(order: CallbackOrder):
    """The method ``on_experiment_start`` is used to initialise \
    a class with method ``Callbacks.on_experiment_start``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_experiment_start, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_experiment_start(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Experiment started")
            #This callback will print "Experiment started" at the start of experiment.

    Returns:
            Initialised callback with method ``Callbacks.on_experiment_start``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_experiment_start)(func, order)

    return decorator


def on_epoch_start(order: CallbackOrder):
    """The method ``on_epoch_start`` is used to initialise \
    a class with method ``Callbacks.on_epoch_start``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_epoch_start, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_epoch_start(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Epoch started")

            #This callback will print "Epoch started" at the start of epoch.

    Returns:
            Initialised callback with method ``Callbacks.on_epoch_start``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_epoch_start)(func, order)

    return decorator


def on_loader_start(order: CallbackOrder):
    """The method ``on_loader_start`` is used to \
    initialise a class with method ``Callbacks.on_loader_start``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_loader_start, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_loader_start(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Loader started")

            #This callback will print "Loader started" at the start of loader.

    Returns:
            Initialised callback with method ``Callbacks.on_loader_start``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_loader_start)(func, order)

    return decorator


def on_batch_start(order: CallbackOrder):
    """The method ``on_batch_start`` is used to \
    initialise a class with method ``Callbacks.on_batch_start``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_batch_start, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_batch_start(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Batch started")

            #This callback will print "Batch started" at the start of batch.

    Returns:
            Initialised callback with method ``Callbacks.on_batch_start``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_batch_start)(func, order)

    return decorator


def on_experiment_end(order: CallbackOrder):
    """The method ``on_experiment_end`` is used to initialise \
    a class with method ``Callbacks.on_experiment_end``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_experiment_end, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_experiment_end(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Experiment Ended")

            #This callback will print "Experiment Ended" at the end of experiment.

    Returns:
            Initialised callback with method ``Callbacks.on_experiment_end``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_experiment_end)(func, order)

    return decorator


def on_epoch_end(order: CallbackOrder):
    """The method ``on_epoch_end`` is used to initialise \
    a class with method ``Callbacks.on_epoch_end``.

    Args:
        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_experiment_end, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_epoch_end(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Epoch Ended")

            #This callback will print "Epoch Ended" at the end of epoch.

    Returns:
            Initialised callback with method ``Callbacks.on_epoch_end``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_epoch_end)(func, order)

    return decorator


def on_loader_end(order: CallbackOrder):
    """The method ``on_loader_end`` is used to initialise \
    a class with method ``Callbacks.on_loader_end``.

    Args:

        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_loader_end, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_loader_end(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("loader Ended")

            #This callback will print "loader Ended" at the end of loader.

    Returns:
            Initialised callback with method ``Callbacks.on_loader_end``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_loader_end)(func, order)

    return decorator


def on_batch_end(order: CallbackOrder):
    """The method ``on_batch_end`` is used to initialise \
    a class with method ``Callbacks.on_batch_end``.

    Args:

        order: The callback order.

    Example:
        .. code-block:: python

            from torchflare.callbacks import on_batch_end, CallbackOrder
            from torchflare.experiments import Experiment

            #Creating a custom callback using the decorator.

            @on_batch_end(order=CallbackOrder.MODEL_INIT)
            def print_on_start(experiment : "Experiment"):
                print("Batch Ended")

            #This callback will print "Batch Ended" at the end of batch.

    Returns:
            Initialised callback with method ``Callbacks.on_batch_end``.
    """

    @functools.wraps(order)
    def decorator(func):
        return bind_to_event(Callbacks.on_batch_end)(func, order)

    return decorator

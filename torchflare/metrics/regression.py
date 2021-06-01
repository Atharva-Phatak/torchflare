"""Implements regression Metrics."""
import torch

from torchflare.metrics.meters import MetricMeter


def _check_shape(outputs: torch.Tensor, targets: torch.Tensor):
    """Check that predictions and target have the same shape, else raise error.

    Raises:
        ValueError: If shapes of outputs and targets is not same.

    Args:
        outputs: torch.Tensor
        targets: torch.Tensor
    """
    if outputs.shape != targets.shape:
        raise ValueError("Predictions and targets are expected to have the same shape")


def detach_tensor(x):
    """Detaches tensor from the graph and moves to cpu.

    Args:
        x:torch.Tensor

    Returns:
        torch.Tensor
    """
    return x.detach().cpu()


class MAE(MetricMeter):
    """Computes Mean Absolute Error."""

    def __init__(self):
        """Constructor method for MAE."""
        self._n_obs = None
        self._abs_error_sum = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The name of the class
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs, targets):
        """Accumulates the batch outputs and targets.

        Args:
            outputs(torch.Tensor): raw logits from the network.
            targets(torch.Tensor) : targets to use for computing accuracy
        """
        outputs, targets = detach_tensor(outputs), detach_tensor(targets)
        _check_shape(outputs, targets)
        self._abs_error_sum += torch.sum(torch.abs(outputs - targets))
        self._n_obs += targets.numel()

    def reset(self):
        """Reset the output and target lists."""
        self._n_obs = torch.tensor(0)
        self._abs_error_sum = torch.tensor(0.0)

    @property
    def value(self) -> torch.Tensor:
        """Computes the MAE.

        Returns:
            The computed MAE.
        """
        return self._abs_error_sum / self._n_obs


class MSE(MetricMeter):
    """Computes Mean Squared Error."""

    def __init__(self):
        """Constructor Method for MSE."""
        self._n_obs = None
        self._squared_error_sum = None

        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The name of the class
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs, targets):
        """Accumulates the batch outputs and targets.

        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : targets to use for computing accuracy
        """
        outputs, targets = detach_tensor(outputs), detach_tensor(targets)
        _check_shape(outputs, targets)
        self._squared_error_sum += torch.sum(torch.pow(outputs - targets, 2))
        self._n_obs += targets.numel()

    def reset(self):
        """Reset the output and target lists."""
        self._n_obs = torch.tensor(0)
        self._squared_error_sum = torch.tensor(0.0)

    @property
    def value(self) -> torch.Tensor:
        """Computes the MSE.

        Returns:
            The computed MSE.
        """
        return self._squared_error_sum / self._n_obs


class MSLE(MetricMeter):
    """Computes Mean Squared Log Error."""

    def __init__(self):
        """Constructor Method for MSLE."""
        self._n_obs = None
        self._log_squared_error_sum = None
        self.reset()

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The name of the class
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs, targets):
        """Accumulates the batch outputs and targets.

        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : targets to use for computing accuracy
        """
        outputs, targets = detach_tensor(outputs), detach_tensor(targets)
        _check_shape(outputs, targets)
        diff = torch.log1p(outputs) - torch.log1p(targets)
        self._log_squared_error_sum += torch.sum(torch.pow(diff, 2))
        self._n_obs += targets.numel()

    def reset(self):
        """Reset the output and target lists."""
        self._n_obs = torch.tensor(0)
        self._log_squared_error_sum = torch.tensor(0.0)

    @property
    def value(self) -> torch.Tensor:
        """Computes the MSLE.

        Returns:
            The computed MSLE.
        """
        return self._log_squared_error_sum / self._n_obs


class R2Score(MetricMeter):
    """Computes R2-score."""

    def __init__(self):
        """Constructor method for R2-score."""
        self._num_examples = None
        self._sum_of_errors = None
        self._y_sq_sum = None
        self._y_sum = None
        self.reset()

    def reset(self) -> None:
        """Reset the output and target lists."""
        self._num_examples = 0
        self._sum_of_errors = torch.tensor(0.0)
        self._y_sq_sum = torch.tensor(0.0)
        self._y_sum = torch.tensor(0.0)

    def handle(self) -> str:
        """Method to get the class name.

        Returns:
            The name of the class
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs, targets):
        """Accumulates the batch outputs and targets.

        Args:
            outputs(torch.Tensor) : raw logits from the network.
            targets(torch.Tensor) : targets to use for computing accuracy
        """
        self._num_examples += outputs.shape[0]
        self._sum_of_errors += torch.sum(torch.pow(outputs - targets, 2))

        self._y_sum += torch.sum(targets)
        self._y_sq_sum += torch.sum(torch.pow(targets, 2))

    @property
    def value(self) -> torch.Tensor:
        """Computes the R2Score.

        Raises:
            ValueError:  If no examples are found.

        Returns:
            The computed R2Score.
        """
        if self._num_examples == 0:
            raise ValueError("R2Score must have at least one example before it can be computed.")
        return 1 - self._sum_of_errors / (self._y_sq_sum - (self._y_sum ** 2) / self._num_examples)


__all__ = ["MAE", "MSE", "MSLE", "R2Score"]

"""Base class implementation for CAM methods."""
import torch

from torchflare.interpreters.gradients import SaveHooks


class BaseCam:
    """Base class for CAM based algorithms."""

    def __init__(self, model, target_layer):
        """Constructor for BaseCam Class.

        Args:
            model : The model
            target_layer: The layer to be used for calculating CAMs.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.values = SaveHooks(layer=self.target_layer)

    def _get_cam_data(self, values, score, target_category):
        """Method to get cam data."""
        raise NotImplementedError

    def forward(self, input_tensor: torch.Tensor, target_category: int) -> torch.Tensor:
        """Forward Pass.

        Args:
            input_tensor: The input tensor to the model.
            target_category: The target category for the input tensor.

        Returns:
            class activation mapping.
        """
        score = self.model(input_tensor)
        cam = self._get_cam_data(values=self.values, score=score, target_category=target_category)
        return cam

    def __call__(self, input_tensor, target_category) -> torch.Tensor:
        """__call__ method.

        Args:
            input_tensor: The input tensor to the model.
            target_category: The target category for the input tensor.

        Returns
            class activation map
        """
        return self.forward(input_tensor=input_tensor, target_category=target_category)

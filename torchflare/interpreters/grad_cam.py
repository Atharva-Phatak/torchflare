"""Implementation of GradCam."""
from abc import ABC

import torch
import torch.nn.functional as F

from torchflare.interpreters.base_cam import BaseCam


class GradCam(BaseCam, ABC):
    """Implementation of `GradCam <https://arxiv.org/abs/1610.02391>`_.

    Args:
            model: The model to be used for gradcam.
            target_layer: The target layer to be used for cam extraction.

    Examples:

        .. code-block:: python

            from torchflare.interpreters import GradCam, visualize_cam

            cam_model = GradCam(model=model, target_layer=target_layer)
            cam = cam_model(tensor, target_category=282)
            visualize_cam(image=image, cam=cam)
    """

    def __init__(self, model, target_layer):
        """Constructor method for GradCam."""
        super(GradCam, self).__init__(model=model, target_layer=target_layer)

    def _get_cam_data(self, values, score, target_category):

        self.model.zero_grad()
        score[0, target_category].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients

        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data

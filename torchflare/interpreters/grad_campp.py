"""Implementation of GradCam++."""
from abc import ABC

import torch
import torch.nn.functional as F

from torchflare.interpreters.base_cam import BaseCam


class GradCamPP(BaseCam, ABC):
    """Implementation of `GradCam++` _.

    .. _GradCam++: https://arxiv.org/pdf/1710.11063.pdf

    Args:
            model: The model to be used for gradcam.
            target_layer: The target layer to be used for cam extraction.

    Examples:

        .. code-block:: python

            from torchflare.interpreters import GradCamPP, visualize_cam

            cam_model = GradCamPP(model=model, target_layer=target_layer)
            cam = cam_model(tensor, target_category=282)
            visualize_cam(image=image, cam=cam)
    """

    def __init__(self, model, target_layer):
        """Constructor method for GradCam."""
        super(GradCamPP, self).__init__(model=model, target_layer=target_layer)

    def _get_cam_data(self, values, score, target_category):

        self.model.zero_grad()
        score[0, target_category].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients

        n, c, _, _ = gradients.shape

        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, target_category].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data

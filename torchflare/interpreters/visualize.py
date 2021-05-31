"""Implementation of visualization method."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_cam(image: np.ndarray, cam: torch.Tensor, alpha: float = 0.6):
    """Method to visualize the generated cam superimposed on image.

    Args:
        image(numpy array): The image converted to numpy array
        cam(torch.Tensor): The class activation map tensor.
        alpha(float): weight for input image for transperancy/blending.
    """
    heatmap = cam.squeeze().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    plt.imshow(heatmap)


__all__ = ["visualize_cam"]

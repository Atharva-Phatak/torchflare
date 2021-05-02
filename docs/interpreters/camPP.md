::: torchflare.interpreters.grad_campp.GradCamPP
    rendering:
             show_root_toc_entry: false

## Example
``` python
from torchflare.interpreters import GradCamPP, visualize_cam

cam_model = GradCamPP(model=model, target_layer=target_layer)
cam = cam_model(tensor, target_category=282)
visualize_cam(image=image, cam=cam)
```

#flake8: noqa
import torch

from torchflare.interpreters.grad_cam import GradCam
from torchflare.interpreters.grad_campp import GradCamPP
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# Some basic stuff required for both tests.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), normalize])
image = Image.open("tests/interpreters/tigercat.jpg")
# convert image to tensor
tensor = preprocess(image)

# reshape 4D tensor (N, C, H, W)
tensor = tensor.unsqueeze(0)
model = torchvision.models.resnet18(pretrained=True)
target_layer = model.layer4[1].conv2


def test_gradcam():
    cam_model = GradCam(model=model, target_layer=target_layer)
    cam = cam_model(tensor, target_category=282)

    assert torch.is_tensor(cam) is True
    assert len(cam.shape) == 4


def test_gradcamPP():
    cam_model = GradCamPP(model=model, target_layer=target_layer)
    cam = cam_model(tensor, target_category=282)

    assert torch.is_tensor(cam) is True
    assert len(cam.shape) == 4

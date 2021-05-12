"""Implements Utility functions for datasets."""
from typing import Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms as transforms


def DecodeRLE(mask_rle: str, shape: Tuple):
    """A function to decode run length encoding.

    Args:
        mask_rle : rule length encoding
        shape : A tuple for the specified shape for mask
                Ex : shape = (128 , 128)

    Returns:
        Decoded encoding as numpy array.
    """
    # print(type(mask_rle))
    s = mask_rle.split()
    start, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    end = start + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo_idx, hi_idx in zip(start, end):
        img[lo_idx:hi_idx] = 1

    return img.reshape(shape, order="F")


def EncodeRLE(mask):
    """Convert mask to EncodedPixels in run-length encoding.

    Source : https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines

    Args:
        mask : the mask array which you want to convert to Rule Length Encoding.

    Returns:
        RLE for the mask.
    """
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def make_masks(rle, shape, num_classes):
    """Create masks based on RLE.

    Args:
        rle : rule length encoding
        shape : A tuple for the specified shape for mask
                Ex : shape = (128 , 128)
        num_classes : the number of masks if you have a multiclasss segmentation problem

    Returns:
        numpy array of masks.
    """
    masks = np.zeros((shape[0], shape[1], num_classes), dtype=np.float32)
    # print(len(rle))

    for idx, label in enumerate(rle):
        if isinstance(label, (list, tuple)):
            label = label[0]
        if isinstance(label, str):
            mask = DecodeRLE(label, shape=shape)
            masks[:, :, idx] = mask

    return masks


def handle_shape(x):
    """Handles shape for albumentations augs.

    Args:
        x : the input image.

    Returns:
        The image after handling the shape.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.expand_dims(x, 2)
    x = np.transpose(x, (1, 0, 2))
    x = np.transpose(x, (2, 1, 0))

    return x


def open_image(x, convert_mode="RGB"):
    """Opens Image.

    Args:
        x : The path to image.
        convert_mode: The mode in which image has to be opened.

    Returns:
        PIL.Image in the required mode.
    """
    x = PIL.Image.open(x).convert(convert_mode)
    return x


def apply_image_transforms(image, augs) -> torch.Tensor:
    """Apply augmentations to image.

    Args:
        image: The input image.
        augs: The augmentations to be applied.
    """
    if augs is None:
        return to_tensor(image)

    if isinstance(augs, A.Compose):
        image = np.array(image)
        augmented_image = augs(image=image)
        image = augmented_image.get("image")
        image = handle_shape(image)
        return to_tensor(image)

    else:
        image = augs(image)
        return to_tensor(image)


def apply_albu_augs(image, augs, mask):
    """Applies albumentations augmentations.

    Args:
        image: The input image.
        augs: The albumentations augmentations.
        mask: The mask if problem type is image segmentation,

    Returns:
        image , mask if image segmentation else returns image.
    """
    image = np.array(image)
    mask = np.array(mask)
    augmented_image = augs(image=image, mask=mask)
    image = augmented_image.get("image")
    mask = augmented_image.get("mask")
    return image, mask


def apply_torchvision_augs(image, augs, mask):
    """Applies albumentations augmentations.

    Args:
        image: The input image.
        augs: The torchvision augmentations.
        mask: The mask if problem type is image segmentation,

    Returns:
        image , mask if image segmentation else returns image.
    """
    image = augs(image)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
        mask = torchvision.transforms.ToPILImage()(mask)

    mask = augs(mask)
    return image, mask


def apply_segmentation_augs(image, augs, mask=None):
    """Apply albumentations/torchvision augmentations for segmentation task.

    Args:
        image: The input image.
        augs: The augmentations.
        mask : The input mask(only applicable for image segmentation problems.)

    Returns:
        Image , mask if problem is image segmentation else returns image
    """
    if isinstance(augs, A.Compose):

        image, mask = apply_albu_augs(image=image, augs=augs, mask=mask)
        image = handle_shape(image)
        mask = handle_shape(mask)

    elif isinstance(augs, torchvision.transforms.Compose):

        image, mask = apply_torchvision_augs(image=image, augs=augs, mask=mask)

    return to_tensor(image), to_tensor(mask)


def to_tensor(x):
    """Converts input to tensor.

    Args:
        x : input

    Returns:
        Torch.tensor
    """
    if not isinstance(x, torch.Tensor) and isinstance(x, PIL.Image.Image):

        return transforms.functional.to_tensor(x)
    elif not isinstance(x, torch.Tensor) and x is not None:
        return torch.tensor(x)
    else:
        return x


def show_batch(dl, **kwargs):
    """Method to visualize the batch for image data.

    Args:
        dl : The pytorch dataloader.
        **kwargs: keyword arguments for torchvision.utils.make_grid function.

    Note:
        Only use for classification and segmentations tasks.
    """
    op = next(iter(dl))
    if isinstance(op, (list, tuple)):
        x, _ = op
    else:
        x = op
    grid_images = torchvision.utils.make_grid(x, **kwargs)
    grid_images = grid_images.numpy()
    plt.imshow(np.transpose(grid_images, (1, 2, 0)), interpolation="nearest")

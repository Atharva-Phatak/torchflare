import glob
import os
from typing import Tuple

import numpy
import PIL
import torch
import torchvision


def get_iloc_cols(df, cols):
    """Get values of columns from dataframe.

    Args:
        df: pandas dataframe.
        cols: The columns for which value is to be extracted.
    """
    columns = [df.columns.get_loc(c) for c in cols]
    iloc_cols = columns[0] if len(columns) == 1 else columns
    return df.iloc[:, iloc_cols].values.tolist()


def is_none(obj):
    """Method to check if object is None.

    Args:
        obj : The object.
    """
    return bool(obj is None)


def get_labels_from_paths(target_path):
    """Method to get labels from folders.

    Args:
        target_path: The path where input folders are stored.
    """
    paths = glob.glob(target_path + "/*")
    labels = [p.split(os.path.sep)[-1] for p in paths]
    labels_to_idx = {label: i for i, label in enumerate(set(labels))}
    return labels_to_idx


def get_class_to_idx(item, class_mapping):
    """Method to return id for corresponding item."""
    for k in class_mapping:
        if k in item:
            return class_mapping[k]


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


def to_tensor(x, dtype=None):
    """Converts input to tensor.

    Args:
        x : input
        dtype : The data type for tensor.

    Returns:
        Torch.tensor
    """
    if not isinstance(x, torch.Tensor) and isinstance(x, PIL.Image.Image):
        op = torchvision.transforms.functional.to_tensor(x)
    elif not isinstance(x, torch.Tensor) and x is not None:
        op = torch.tensor(x)
    elif isinstance(x, numpy.ndarray):
        op = torch.from_numpy(x)
    else:
        op = x
    if dtype is not None:
        return op.type(dtype)
    return op


def join_paths(path, files, extension):
    """Method to merge paths."""
    if extension is not None:
        return [path / (x + extension) for x in files]
    return [path / x for x in files]


def handle_shape(x):
    """Handles shape for albumentations augs.

    Args:
        x : the input image.

    Returns:
        The image after handling the shape.
    """
    x = numpy.asarray(x)
    if x.ndim == 2:
        x = numpy.expand_dims(x, 2)
    x = numpy.transpose(x, (1, 0, 2))
    x = numpy.transpose(x, (2, 1, 0))

    return x


def apply_image_augmentations(image, transforms):
    """Apply albumentations augmentations on image.

    Args:
        image: The input image.
        transforms: The albumentations augmentations.

    Returns:
        transformed image
    """
    image = numpy.array(image)
    image = transforms(image=image)["image"]
    image = handle_shape(image)
    return to_tensor(image)


def apply_segmentation_augs(image, mask, transforms):
    """Applies albumentations augmentations to segmentation problem.

    Args:
        image: The input image.
        transforms: The albumentations augmentations.
        mask: The mask if problem type is image segmentation,

    Returns:
        image , mask
    """
    image, mask = numpy.array(image), numpy.array(mask)
    augmented_image = transforms(image=image, mask=mask)
    image = augmented_image.get("image")
    mask = augmented_image.get("mask")
    image, mask = handle_shape(image), handle_shape(mask)
    return to_tensor(image), to_tensor(mask)


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
    start, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    end = start + lengths
    img = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
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
        pixel_padded = numpy.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = numpy.where(pixels[1:] != pixels[:-1])[0] + 2
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
    masks = numpy.zeros((shape[0], shape[1], num_classes), dtype=numpy.float32)
    # print(len(rle))

    for idx, label in enumerate(rle):
        if isinstance(label, (list, tuple)):
            label = label[0]
        if isinstance(label, str):
            mask = DecodeRLE(label, shape=shape)
            masks[:, :, idx] = mask

    return masks

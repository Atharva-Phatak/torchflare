# flake8: noqa
import collections

import albumentations as A
import pandas as pd
import torch
import torchvision

from torchflare.datasets.segmentation import SegmentationDataset
from torchflare.datasets.dataloaders import SimpleDataloader

df = pd.read_csv("tests/datasets/data/image_segmentation/csv_data/dummy.csv")
df_inputs = collections.namedtuple(
    "df_inputs", ["path", "image_col", "mask_cols", "mask_size", "num_classes", "extension", "df"],
)
df_inputs = df_inputs(
    path="tests/datasets/data/image_segmentation/csv_data/images",
    df=df,
    image_col="im_id",
    mask_cols=["EncodedPixels"],
    extension=None,
    mask_size=(320, 320),
    num_classes=4,
)

folder_inputs = collections.namedtuple("folder_inputs", ["image_path", "mask_path"])
folder_inputs = folder_inputs(
    image_path="tests/datasets/data/image_segmentation/folder_data/images",
    mask_path="tests/datasets/data/image_segmentation/folder_data/masks",
)


def test_from_df():
    def test_augmentations_augs():
        augmentations = A.Compose([A.Resize(256, 256)])

        ds = SegmentationDataset.from_rle(
            path=df_inputs.path,
            df=df,
            image_col=df_inputs.image_col,
            mask_cols=df_inputs.mask_cols,
            extension=df_inputs.extension,
            mask_size=df_inputs.mask_size,
            num_classes=df_inputs.num_classes,
            augmentations=augmentations,
            image_convert_mode="RGB",
        )

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (3, 256, 256)
        assert y.shape == (df_inputs.num_classes, 256, 256)

    def test_torchvision_augs():

        augmentations = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256))])

        ds = SegmentationDataset.from_rle(
            path=df_inputs.path,
            df=df,
            image_col=df_inputs.image_col,
            mask_cols=df_inputs.mask_cols,
            extension=df_inputs.extension,
            mask_size=df_inputs.mask_size,
            num_classes=df_inputs.num_classes,
            augmentations=augmentations,
            image_convert_mode="RGB",
        )

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (3, 256, 256)
        assert y.shape == (df_inputs.num_classes, 256, 256)

    def test_inference():

        ds = SegmentationDataset.from_rle(
            path=df_inputs.path,
            df=df,
            image_col=df_inputs.image_col,
            mask_cols=None,
            extension=df_inputs.extension,
            mask_size=None,
            num_classes=None,
            augmentations=None,
            image_convert_mode="RGB",
        )
        x = ds[0]

        assert torch.is_tensor(x) is True
        assert len(x.shape) == 3

    test_torchvision_augs()
    test_augmentations_augs()
    test_inference()


# test_from_df()


def test_from_folders():
    def test_albu_augs():
        augmentations = A.Compose([A.Resize(256, 256)])

        ds = SegmentationDataset.from_folders(
            image_path=folder_inputs.image_path,
            mask_path=folder_inputs.mask_path,
            augmentations=augmentations,
            image_convert_mode="L",
            mask_convert_mode="L",
        )

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (1, 256, 256)
        assert y.shape == (1, 256, 256)

    def test_torchvision_augs():

        augmentations = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256))])
        ds = SegmentationDataset.from_folders(
            image_path=folder_inputs.image_path,
            mask_path=folder_inputs.mask_path,
            augmentations=augmentations,
            image_convert_mode="L",
            mask_convert_mode="L",
        )

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (1, 256, 256)
        assert y.shape == (1, 256, 256)

    test_albu_augs()
    test_torchvision_augs()


def test_segmentation_dataloaders():
    def test_segmentation_data_from_rle():
        augmentations = A.Compose([A.Resize(256, 256)])

        dl = SimpleDataloader.segmentation_data_from_rle(
            path=df_inputs.path,
            df=df,
            image_col=df_inputs.image_col,
            mask_cols=df_inputs.mask_cols,
            extension=df_inputs.extension,
            mask_size=df_inputs.mask_size,
            num_classes=df_inputs.num_classes,
            augmentations=augmentations,
            image_convert_mode="RGB",
        ).get_loader(batch_size=2, shuffle=True)

        x, y = next(iter(dl))

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (2, 3, 256, 256)
        assert y.shape == (2, df_inputs.num_classes, 256, 256)

    def test_segmentation_data_from_folders():
        augmentations = A.Compose([A.Resize(256, 256)])

        dl = SimpleDataloader.segmentation_data_from_folders(
            image_path=folder_inputs.image_path,
            mask_path=folder_inputs.mask_path,
            augmentations=augmentations,
            image_convert_mode="L",
            mask_convert_mode="L",
        ).get_loader(batch_size=2, shuffle=False)

        x, y = next(iter(dl))

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (2, 1, 256, 256)
        assert y.shape == (2, 1, 256, 256)

    test_segmentation_data_from_folders()
    test_segmentation_data_from_rle()

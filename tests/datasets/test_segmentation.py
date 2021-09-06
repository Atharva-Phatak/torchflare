# flake8 : noqa
import collections

import albumentations as A
import pandas as pd
import torch
from torchflare.datasets import SegmentationDataset
import pytest

df_inputs = collections.namedtuple(
    "df_inputs",
    ["path", "image_col", "mask_cols", "mask_size", "num_classes", "extension", "df", "augmentations"],
)
df_inputs = df_inputs(
    path="tests/datasets/data/image_segmentation/csv_data/images",
    df=pd.read_csv("tests/datasets/data/image_segmentation/csv_data/dummy.csv"),
    image_col=["im_id"],
    mask_cols=["EncodedPixels"],
    extension=None,
    mask_size=(320, 320),
    num_classes=4,
    augmentations=A.Compose([A.Resize(256, 256)]),
)

folder_inputs = collections.namedtuple(
    "folder_inputs", ["image_path", "mask_path", "image_convert_mode", "mask_convert_mode", "augmentations"]
)
folder_inputs = folder_inputs(
    image_path="tests/datasets/data/image_segmentation/folder_data/images",
    mask_path="tests/datasets/data/image_segmentation/folder_data/masks",
    image_convert_mode="L",
    mask_convert_mode="L",
    augmentations=A.Compose([A.Resize(256, 256)]),
)


class TestSegmentationDataFromDF:
    def test_with_input_transforms(self):
        ds = SegmentationDataset.from_df(
            df=df_inputs.df,
            path=df_inputs.path,
            input_columns=df_inputs.image_col,
            transforms=df_inputs.augmentations,
        ).masks_from_rle(
            shape=df_inputs.mask_size,
            num_classes=df_inputs.num_classes,
            mask_columns=df_inputs.mask_cols,
        )
        x, y = ds[0]
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (3, 256, 256)
        assert y.shape == (df_inputs.num_classes, 256, 256)

    def test_inference(self):
        ds = SegmentationDataset.from_df(
            df=df_inputs.df,
            path=df_inputs.path,
            input_columns=df_inputs.image_col,
            transforms=df_inputs.augmentations,
        ).add_test()

        x = ds[0]
        assert torch.is_tensor(x) is True
        assert len(x.shape) == 3

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_batching(self, batch_size):
        dl = (
            SegmentationDataset.from_df(
                df=df_inputs.df,
                path=df_inputs.path,
                input_columns=df_inputs.image_col,
                transforms=df_inputs.augmentations,
            )
            .masks_from_rle(
                shape=df_inputs.mask_size,
                num_classes=df_inputs.num_classes,
                mask_columns=df_inputs.mask_cols,
            )
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(dl))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(x) is True
        assert x.shape == (batch_size, 3, 256, 256)
        assert y.shape == (batch_size, df_inputs.num_classes, 256, 256)


class TestSegmentationDataFromFolders:
    def test_with_input_transforms(self):
        ds = SegmentationDataset.from_folders(
            image_path=folder_inputs.image_path,
            transforms=folder_inputs.augmentations,
            image_convert_mode=folder_inputs.image_convert_mode,
        ).masks_from_folders(
            mask_path=folder_inputs.mask_path,
            mask_convert_mode=folder_inputs.mask_convert_mode,
        )
        x, y = ds[0]
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (1, 256, 256)
        assert y.shape == (1, 256, 256)

    def test_inference(self):
        ds = SegmentationDataset.from_folders(
            image_path=folder_inputs.image_path,
            transforms=folder_inputs.augmentations,
            image_convert_mode=folder_inputs.image_convert_mode,
        ).add_test()
        x = ds[0]
        assert torch.is_tensor(x) is True
        assert x.shape == (1, 256, 256)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_batching(self, batch_size):
        dl = (
            SegmentationDataset.from_folders(
                image_path=folder_inputs.image_path,
                transforms=folder_inputs.augmentations,
                image_convert_mode=folder_inputs.image_convert_mode,
            )
            .masks_from_folders(
                mask_path=folder_inputs.mask_path,
                mask_convert_mode=folder_inputs.mask_convert_mode,
            )
            .batch(batch_size=batch_size, shuffle=True)
        )
        x, y = next(iter(dl))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, 1, 256, 256)
        assert y.shape == (batch_size, 1, 256, 256)

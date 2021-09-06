# flake8 : noqa
import collections
import albumentations as A
import pandas as pd
import torch
from torchflare.datasets import ImageDataset
import pytest
from functools import partial

inputs = collections.namedtuple(
    "inputs",
    [
        "path",
        "csv_path",
        "extension",
        "augmentations",
        "label_cols",
        "df",
        "image_col",
        "convert_mode",
    ],
)
constants_df = inputs(
    path="tests/datasets/data/image_classification/csv_data/images",
    csv_path="tests/datasets/data/image_classification/csv_data/train.csv",
    df=pd.read_csv("tests/datasets/data/image_classification/csv_data/train.csv"),
    extension=".jpg",
    label_cols=["healthy", "multiple_diseases", "rust", "scab"],
    image_col=["image_id"],
    augmentations=A.Compose([A.Resize(256, 256)]),
    convert_mode="RGB",
)

folder_inputs = collections.namedtuple("folder_inputs", ["train_path", "test_path", "augmentations"])
folder_inputs = folder_inputs(
    train_path="tests/datasets/data/image_classification/folder_data/train_data",
    test_path="tests/datasets/data/image_classification/folder_data/test_data",
    augmentations=A.Compose([A.Resize(256, 256)]),
)


class TestImageDatasetDF:
    @pytest.mark.parametrize("input_transforms", [constants_df.augmentations, None])
    def test_with_input_transforms(self, input_transforms):
        ds = ImageDataset.from_df(
            path=constants_df.path,
            df=constants_df.df,
            transforms=input_transforms,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).targets_from_df(target_columns=constants_df.label_cols)

        x, y = ds[0]

        if input_transforms is not None:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True
            assert x.shape == (3, 256, 256)
            assert y.shape[0] == 4
        else:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True
            assert y.shape[0] == 4

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = ImageDataset.from_df(
            path=constants_df.path,
            df=constants_df.df,
            transforms=constants_df.augmentations,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).targets_from_df(target_columns=constants_df.label_cols, target_transforms=target_transforms)
        x, y = ds[0]

        assert torch.is_tensor(y)
        if target_transforms is not None:
            assert y.dtype == torch.float

    def test_for_inference(self):
        ds = ImageDataset.from_df(
            path=constants_df.path,
            df=constants_df.df,
            transforms=constants_df.augmentations,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).add_test()

        x = ds[0]
        assert torch.is_tensor(x)
        assert x.shape == (3, 256, 256)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        dl = (
            ImageDataset.from_df(
                path=constants_df.path,
                df=constants_df.df,
                transforms=constants_df.augmentations,
                input_columns=constants_df.image_col,
                extension=constants_df.extension,
                convert_mode=constants_df.convert_mode,
            )
            .targets_from_df(target_columns=constants_df.label_cols)
            .batch(batch_size=batch_size, shuffle=True)
        )
        x, y = next(iter(dl))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, 3, 256, 256)
        assert y.shape == (batch_size, 4)


class TestImageDatasetFolders:
    @pytest.mark.parametrize("input_transforms", [folder_inputs.augmentations, None])
    def test_with_input_transforms(self, input_transforms):
        ds = ImageDataset.from_folders(
            path=folder_inputs.train_path,
            transforms=input_transforms,
            convert_mode="RGB",
        ).targets_from_folders(target_path=folder_inputs.train_path)

        x, y = ds[0]

        if input_transforms is not None:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True
            assert x.shape == (3, 256, 256)
        else:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = ImageDataset.from_folders(
            path=folder_inputs.train_path,
            transforms=folder_inputs.augmentations,
            convert_mode="RGB",
        ).targets_from_folders(target_path=folder_inputs.train_path, target_transforms=target_transforms)
        x, y = ds[0]

        assert torch.is_tensor(y)
        if target_transforms is not None:
            assert y.dtype == torch.float

    def test_for_inference(self):
        ds = ImageDataset.from_folders(
            path=folder_inputs.train_path,
            transforms=folder_inputs.augmentations,
            convert_mode="RGB",
        ).add_test()

        x = ds[0]
        assert torch.is_tensor(x)
        assert x.shape == (3, 256, 256)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        dl = (
            ImageDataset.from_folders(
                path=folder_inputs.train_path,
                transforms=folder_inputs.augmentations,
                convert_mode="RGB",
            )
            .targets_from_folders(target_path=folder_inputs.train_path)
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(dl))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, 3, 256, 256)
        assert y.shape[0] == batch_size


class TestImageDataset_CSV:
    @pytest.mark.parametrize("input_transforms", [constants_df.augmentations, None])
    def test_with_input_transforms(self, input_transforms):
        ds = ImageDataset.from_csv(
            path=constants_df.path,
            csv_path=constants_df.csv_path,
            transforms=input_transforms,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).targets_from_df(target_columns=constants_df.label_cols)

        x, y = ds[0]

        if input_transforms is not None:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True
            assert x.shape == (3, 256, 256)
            assert y.shape[0] == 4
        else:
            assert torch.is_tensor(x) is True
            assert torch.is_tensor(y) is True
            assert y.shape[0] == 4

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = ImageDataset.from_csv(
            path=constants_df.path,
            csv_path=constants_df.csv_path,
            transforms=constants_df.augmentations,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).targets_from_df(target_columns=constants_df.label_cols, target_transforms=target_transforms)
        x, y = ds[0]

        assert torch.is_tensor(y)
        if target_transforms is not None:
            assert y.dtype == torch.float

    def test_for_inference(self):
        ds = ImageDataset.from_csv(
            path=constants_df.path,
            csv_path=constants_df.csv_path,
            transforms=constants_df.augmentations,
            input_columns=constants_df.image_col,
            extension=constants_df.extension,
            convert_mode=constants_df.convert_mode,
        ).add_test()

        x = ds[0]
        assert torch.is_tensor(x)
        assert x.shape == (3, 256, 256)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        dl = (
            ImageDataset.from_csv(
                path=constants_df.path,
                csv_path=constants_df.csv_path,
                transforms=constants_df.augmentations,
                input_columns=constants_df.image_col,
                extension=constants_df.extension,
                convert_mode=constants_df.convert_mode,
            )
            .targets_from_df(target_columns=constants_df.label_cols)
            .batch(batch_size=batch_size, shuffle=True)
        )
        x, y = next(iter(dl))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, 3, 256, 256)
        assert y.shape == (batch_size, 4)

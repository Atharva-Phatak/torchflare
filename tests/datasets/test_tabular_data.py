# flake8 : noqa
import collections
import pandas as pd
import torch
from torchflare.datasets import TabularDataset
import pytest
from functools import partial

inputs = collections.namedtuple("inputs", ["df", "csv_path", "input_columns", "target_columns"])
df = pd.read_csv("tests/datasets/data/tabular_data/diabetes.csv")
tabular_inputs = inputs(
    df=df,
    csv_path="tests/datasets/data/tabular_data/diabetes.csv",
    target_columns=["Outcome"],
    input_columns=[col for col in df.columns if col not in ["Outcome"]],
)


class TestTabularDF:
    @pytest.mark.parametrize("input_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_input_transforms(self, input_transforms):
        ds = TabularDataset.from_df(
            df=tabular_inputs.df,
            input_columns=tabular_inputs.input_columns,
            transforms=input_transforms,
        ).targets_from_df(target_columns=tabular_inputs.target_columns)
        x, y = ds[0]

        if input_transforms is not None:
            assert x.dtype == torch.float
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = TabularDataset.from_df(
            df=tabular_inputs.df,
            input_columns=tabular_inputs.input_columns,
            transforms=None,
        ).targets_from_df(
            target_columns=tabular_inputs.target_columns,
            target_transforms=target_transforms,
        )
        x, y = ds[0]

        if target_transforms is not None:
            assert y.dtype == torch.float
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    def test_with_inference(self):
        ds = TabularDataset.from_df(
            df=tabular_inputs.df,
            input_columns=tabular_inputs.input_columns,
            transforms=None,
        ).add_test()
        x = ds[0]

        assert torch.is_tensor(x) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        ds = (
            TabularDataset.from_df(
                df=tabular_inputs.df,
                input_columns=tabular_inputs.input_columns,
                transforms=None,
            )
            .targets_from_df(target_columns=tabular_inputs.target_columns, target_transforms=None)
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(ds))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, len(tabular_inputs.input_columns))


class TestTabularCSV:
    @pytest.mark.parametrize("input_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_input_transforms(self, input_transforms):
        ds = TabularDataset.from_csv(
            csv_path=tabular_inputs.csv_path,
            input_columns=tabular_inputs.input_columns,
            transforms=input_transforms,
        ).targets_from_df(target_columns=tabular_inputs.target_columns)
        x, y = ds[0]

        if input_transforms is not None:
            assert x.dtype == torch.float
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = TabularDataset.from_csv(
            csv_path=tabular_inputs.csv_path,
            input_columns=tabular_inputs.input_columns,
            transforms=None,
        ).targets_from_df(
            target_columns=tabular_inputs.target_columns,
            target_transforms=target_transforms,
        )
        x, y = ds[0]

        if target_transforms is not None:
            assert y.dtype == torch.float
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    def test_with_inference(self):
        ds = TabularDataset.from_csv(
            csv_path=tabular_inputs.csv_path,
            input_columns=tabular_inputs.input_columns,
            transforms=None,
        ).add_test()
        x = ds[0]

        assert torch.is_tensor(x) is True
        assert x.shape[0] == len(tabular_inputs.input_columns)

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        ds = (
            TabularDataset.from_csv(
                csv_path=tabular_inputs.csv_path,
                input_columns=tabular_inputs.input_columns,
                transforms=None,
            )
            .targets_from_df(target_columns=tabular_inputs.target_columns, target_transforms=None)
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(ds))
        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (batch_size, len(tabular_inputs.input_columns))

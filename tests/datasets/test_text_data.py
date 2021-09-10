# flake8 : noqa
from torchflare.datasets.text_data import TextDataset
import transformers
import pandas as pd
import torch
from collections import namedtuple
import pytest
from functools import partial

inputs = namedtuple("inputs", ["df", "path", "tokenizer", "max_len", "input_col", "label_col"])
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
data_inputs = inputs(
    df=pd.read_csv("tests/datasets/data/text_classification/train.csv"),
    path="tests/datasets/data/text_classification/train.csv",
    tokenizer=tokenizer,
    max_len=128,
    input_col=["tweet"],
    label_col=["label"],
)


class TestTextDataDF:
    def test_with_input_transforms(self):
        ds = TextDataset.from_df(
            df=data_inputs.df,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).targets_from_df(target_columns=data_inputs.label_col)
        x, y = ds[0]
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = TextDataset.from_df(
            df=data_inputs.df,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).targets_from_df(
            target_columns=data_inputs.label_col,
            target_transforms=target_transforms,
        )
        x, y = ds[0]

        if target_transforms is not None:
            assert y.dtype == torch.float
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    def test_with_inference(self):
        ds = TextDataset.from_df(
            df=data_inputs.df,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).add_test()
        x = ds[0]
        assert isinstance(x, dict) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        ds = (
            TextDataset.from_df(
                df=data_inputs.df,
                input_columns=data_inputs.input_col,
                tokenizer=data_inputs.tokenizer,
                max_len=data_inputs.max_len,
            )
            .targets_from_df(
                target_columns=data_inputs.label_col,
            )
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(ds))
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True


class TestTextDataCSV:
    def test_with_input_transforms(self):
        ds = TextDataset.from_csv(
            csv_path=data_inputs.path,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).targets_from_df(target_columns=data_inputs.label_col)
        x, y = ds[0]
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    @pytest.mark.parametrize("target_transforms", [partial(torch.tensor, dtype=torch.float), None])
    def test_with_target_transforms(self, target_transforms):
        ds = TextDataset.from_csv(
            csv_path=data_inputs.path,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).targets_from_df(
            target_columns=data_inputs.label_col,
            target_transforms=target_transforms,
        )
        x, y = ds[0]

        if target_transforms is not None:
            assert y.dtype == torch.float
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    def test_with_inference(self):
        ds = TextDataset.from_csv(
            csv_path=data_inputs.path,
            input_columns=data_inputs.input_col,
            tokenizer=data_inputs.tokenizer,
            max_len=data_inputs.max_len,
        ).add_test()
        x = ds[0]
        assert isinstance(x, dict) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batching(self, batch_size):
        ds = (
            TextDataset.from_csv(
                csv_path=data_inputs.path,
                input_columns=data_inputs.input_col,
                tokenizer=data_inputs.tokenizer,
                max_len=data_inputs.max_len,
            )
            .targets_from_df(
                target_columns=data_inputs.label_col,
            )
            .batch(batch_size=batch_size, shuffle=True)
        )

        x, y = next(iter(ds))
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

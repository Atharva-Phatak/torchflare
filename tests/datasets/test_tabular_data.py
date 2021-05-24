# flake8: noqa
from torchflare.datasets.tabular import TabularDataset
import pandas as pd
import torch
from torchflare.datasets.tabular_dataloader import TabularDataloader
from torchflare.data_config.tabular_configs import TabularDataConfig

def test_data():
    path = "tests/datasets/data/tabular_data/diabetes.csv"
    df = pd.read_csv(path)
    label_col = "Outcome"
    input_cols = [col for col in df.columns if col != label_col]

    def test_from_df():

        ds = TabularDataset.from_df(df=df, feature_cols=input_cols, label_cols=label_col)

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(input_cols)

        # Inference

        ds = TabularDataset.from_df(df=df, feature_cols=input_cols, label_cols=None)

        x = ds[0]

        assert torch.is_tensor(x) is True
        assert x.shape[0] == len(input_cols)

    def test_from_csv():

        ds = TabularDataset.from_csv(csv_path=path, feature_cols=input_cols, label_cols=label_col)

        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape[0] == len(input_cols)

        # Inference

        ds = TabularDataset.from_csv(csv_path=path, feature_cols=input_cols, label_cols=None)

        x = ds[0]

        assert torch.is_tensor(x) is True
        assert x.shape[0] == len(input_cols)

    def test_dataloaders():

        dl = TabularDataloader.from_df(df=df, feature_cols=input_cols, label_cols=label_col).get_loader(
            batch_size=2, shuffle=True
        )

        x, y = next(iter(dl))

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (2, len(input_cols))

        dl_path = TabularDataloader.from_csv(csv_path=path, feature_cols=input_cols, label_cols=label_col).get_loader(
            batch_size=2, shuffle=False
        )

        x, y = next(iter(dl_path))

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (2, len(input_cols))

    def test_data_configs():
        cfg = TabularDataConfig.from_df(df=df, feature_cols=input_cols, label_cols=label_col)
        ds = cfg.data_method(**cfg.config)
        x, y = ds[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (len(input_cols),)

        cfg_path = TabularDataConfig.from_csv(csv_path=path, feature_cols=input_cols, label_cols=label_col)
        ds_path = cfg_path.data_method(**cfg_path.config)
        x, y = ds_path[0]

        assert torch.is_tensor(x) is True
        assert torch.is_tensor(y) is True
        assert x.shape == (len(input_cols),)

    test_from_df()
    test_from_csv()
    test_dataloaders()
    test_data_configs()

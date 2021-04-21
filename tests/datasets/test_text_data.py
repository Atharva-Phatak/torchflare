from torchflare.datasets.text_dataset import TextClassificationDataset
from torchflare.datasets.dataloaders import SimpleDataloader
import transformers
import pandas as pd
import torch


def test_data():
    path = "tests/datasets/data/text_classification/train.csv"
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv(path)
    max_len = 128

    def test_train():
        ds = TextClassificationDataset.from_df(
            df=df, input_col="tweet", label_cols="label", tokenizer=tokenizer, max_len=max_len
        )

        x, y = ds[0]
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    def test_inference():

        ds = TextClassificationDataset.from_df(
            df=df, input_col="tweet", label_cols=None, tokenizer=tokenizer, max_len=max_len
        )

        x = ds[0]
        assert isinstance(x, dict) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    def test_dataloader():

        dl = SimpleDataloader.text_data_from_df(
            df=df, input_col="tweet", label_cols="label", tokenizer=tokenizer, max_len=max_len
        ).get_loader(batch_size=2 , shuffle=False)

        x, y = next(iter(dl))
        assert isinstance(x, dict) is True
        assert torch.is_tensor(y) is True

        for key, item in x.items():
            assert torch.is_tensor(item) is True

    test_train()
    test_inference()
    test_dataloader()

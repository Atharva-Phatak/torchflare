"""Text classification using Tiny-Bert on IMDB Dataset.
Dataset : https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format?select=Valid.csv
"""

import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
import transformers
from sklearn.model_selection import train_test_split

import torchflare.callbacks as cbs
import torchflare.criterion as crit
from torchflare.datasets import TextDataset
from torchflare.experiments import Experiment, ModelConfig


# Defining The model
class Model(torch.nn.Module):
    def __init__(self, dropout, out_features):

        super(Model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("prajjwal1/bert-tiny", return_dict=False)
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, out_features)

    def forward(self, x):
        _, o_2 = self.bert(**x)

        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output


if __name__ == "__main__":
    # Reading and splitting data.

    df = pd.read_csv("Train.csv")

    train_df, valid_df = train_test_split(df, stratify=df.label, test_size=0.1, random_state=42)
    tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # Creating training and validation dataloaders
    train_dl = (
        TextDataset.from_df(df=train_df, tokenizer=tokenizer, max_len=128, input_columns=["text"])
        .targets_from_df(target_columns=["label"])
        .batch(batch_size=16, shuffle=True)
    )

    valid_dl = (
        TextDataset.from_df(df=valid_df, tokenizer=tokenizer, max_len=128, input_columns=["text"])
        .targets_from_df(target_columns=["label"])
        .batch(batch_size=16, shuffle=True)
    )

    # Defining metrics
    metric_list = [torchmetrics.Accuracy(threshold=0.6)]

    # Defining Callbacks.
    callbacks = [
        cbs.EarlyStopping(monitor="val_accuracy", patience=2, mode="max"),
        cbs.ModelCheckpoint(
            monitor="val_accuracy", mode="max", save_dir="./", file_name="model.bin"
        ),
        cbs.ReduceLROnPlateau(mode="max", patience=2),
    ]

    config = ModelConfig(
        nn_module=Model,
        module_params={"dropout": 0.3, "out_features": 1},
        optimizer="AdamW",
        optimizer_params={"lr": 3e-4},
        criterion=crit.BCEWithLogitsFlat,
    )

    # Compiling and Running Experiment.
    exp = Experiment(
        num_epochs=3,
        fp16=True,
        device="cuda",
        seed=42,
    )

    # Compiling the experiment
    exp.compile_experiment(
        model_config=config,
        callbacks=callbacks,
        metrics=metric_list,
        main_metric="accuracy",
    )

    # Training the models.
    exp.fit_loader(train_dl=train_dl, valid_dl=valid_dl)

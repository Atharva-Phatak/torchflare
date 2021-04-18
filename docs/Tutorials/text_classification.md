# Text Classification Tutorial

Let's learn how to use tinybert and torchflare for text classification.

* ### Importing Libraries
```python

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
import torchflare.criterion as crit
from torchflare.experiments import Experiment
from torchflare.datasets import SimpleDataloader
```

* ### Reading the data
``` python

train_df = pd.read_csv('Train.csv')
valid_df = pd.read_csv('Valid.csv')
test_df = pd.read_csv('Test.csv')
num_classes = 2

```
* ### Defining Model
``` python
class Model(torch.nn.Module):

    def __init__(self , num_classes = 2):

        super(Model , self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "prajjwal1/bert-tiny", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _ , o_2 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        return output

model = Model(num_classes=2)
```
* ### Creating the dataloaders
``` python
tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

train_dl = SimpleDataloader.text_data_from_df(
                        df = train_df,
                        input_col = 'text',
                        label_cols = 'label',
                        tokenizer = tokenizer,
                        max_len = 128).get_loader(batch_size = 16 , shuffle = True)

valid_dl = SimpleDataloader.text_data_from_df(
                        df = valid_df,
                        input_col = 'text',
                        label_cols = 'label',
                        tokenizer = tokenizer,
                        max_len = 128).get_loader(batch_size = 16)

test_dl = SimpleDataloader.text_data_from_df(
                        df = test_df,
                        input_col = 'text',
                        label_cols = None,
                        tokenizer = tokenizer,
                        max_len = 128).get_loader(batch_size = 16 , shuffle = False)
```

* ### Defining callbacks and metrics
``` python
metric_list = [metrics.Accuracy(num_classes=len(classes), multilabel=False)]

callbacks = [
    cbs.EarlyStopping(monitor=acc.handle(), patience=5),
    cbs.ModelCheckpoint(monitor=acc.handle()),
]
```

* ### Setting up the Experiment
``` python

exp = Experiment(
    num_epochs=10,
    save_dir="./models",
    model_name="bert_cls.bin",
    fp16=False,
    using_batch_mixers=False,
    device="cuda",
    compute_train_metrics=True,
    seed=42,
)

# Compiling the experiment
exp.compile_experiment(
    model=model,
    optimizer="Adam",
    optimizer_params=dict(lr=3e-4),
    callbacks=callbacks,
    scheduler="ReduceLROnPlateau",
    scheduler_params=dict(mode="max", patience=5),
    criterion= crit.BCEWithLogitsFlat, # Using BCEWithLogitsFlat since I dont want to handle shapes my outputs and targets.
    metrics=metric_list,
    main_metric="accuracy",
)

# Training the models.
exp.run_experiment(train_dl = train_dl , valid_dl = valid_dl)
```

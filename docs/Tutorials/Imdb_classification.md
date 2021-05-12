
# Text Classfication using TinyBert
* Dataset: <https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format>
***
#### Importing Libraries
``` python
import pandas as pd
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn

import transformers
import torchflare.callbacks as cbs
import torchflare.metrics as metrics
import torchflare.criterion as crit
from torchflare.experiments import Experiment
from torchflare.datasets import TextDataloader

```

#### Reading the data.
``` python
df = pd.read_csv("Train.csv")
```

#### Splitting the dataset into train and validation data.
``` python
train_df , valid_df =train_test_split(df , stratify = df.label,  test_size = 0.1, random_state = 42)
```

#### Defining training and validation dataloaders.
``` python
tokenizer = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

train_dl = TextDataloader.from_df(
                        df = train_df,
                        input_col = 'text',
                        label_cols = 'label',
                        tokenizer = tokenizer,
                        max_len = 128).get_loader(batch_size = 16 , shuffle = True)

valid_dl = TextDataloader.from_df(
                        df = valid_df,
                        input_col = 'text',
                        label_cols = 'label',
                        tokenizer = tokenizer,
                        max_len = 128).get_loader(batch_size = 16)
```



#### Defining Network architecture.
``` python
class Model(torch.nn.Module):

    def __init__(self):

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

model = Model()
```

#### Defining metrics, callbacks and parameters for optimizer.
``` python
metric_list = [metrics.Accuracy(num_classes=2, multilabel=False)]

callbacks = [
    cbs.EarlyStopping(monitor="accuracy", patience=2, mode = "max"),
    cbs.ModelCheckpoint(monitor="accuracy" , mode = "max"),
    cbs.ReduceLROnPlateau(mode = "max" , patience = 2)
]

# I want to define some custom weight decay to model paramters.
# We will use model_params as an argument in optimizer_params to tell torchflare that, hey we are using custom optimizer params for model.
# If model_params arguments is not used, torchflare by default will use model.parameters() as default params to optimizer.
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias"]
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

```


### Defining, compiling and running the experiment.
``` python
exp = Experiment(
    num_epochs=5,
    save_dir="./models",
    model_name="bert_cls.bin",
    fp16=True,
    device="cuda",
    compute_train_metrics=True,
    seed=42,
)

# Compiling the experiment
exp.compile_experiment(
    model=model,
    optimizer="AdamW",
    optimizer_params=dict(model_params = optimizer_parameters, lr=3e-5), # used model_params argument for custom optimizer params.
    callbacks=callbacks,
    criterion= crit.BCEWithLogitsFlat, # Using BCEWithLogitsFlat since I dont want to handle shapes my outputs and targets.
    metrics=metric_list,
    main_metric="accuracy",
)

# Training the models.
exp.fit_loader(train_dl = train_dl , valid_dl = valid_dl)
```


    Epoch: 1/5
    Train: 2250/2250 [=========================]- 88s 39ms/step - train_loss: 0.5132 - train_accuracy: 0.7401
    Valid: 250/250 [=========================]- 7s 28ms/step - val_loss: 0.4119 - val_accuracy: 0.7472

    Epoch: 2/5
    Train: 2250/2250 [=========================]- 106s 47ms/step - train_loss: 0.3944 - train_accuracy: 0.7834
    Valid: 250/250 [=========================]- 9s 35ms/step - val_loss: 0.3906 - val_accuracy: 0.7855

    Epoch: 3/5
    Train: 2250/2250 [=========================]- 120s 53ms/step - train_loss: 0.3471 - train_accuracy: 0.8050
    Valid: 250/250 [=========================]- 9s 37ms/step - val_loss: 0.3641 - val_accuracy: 0.8062

    Epoch: 4/5
    Train: 2250/2250 [=========================]- 130s 58ms/step - train_loss: 0.3032 - train_accuracy: 0.8218
    Valid: 250/250 [=========================]- 9s 38ms/step - val_loss: 0.3798 - val_accuracy: 0.8221

    Epoch: 5/5
    Train: 2250/2250 [=========================]- 136s 60ms/step - train_loss: 0.2626 - train_accuracy: 0.8349
    Valid: 250/250 [=========================]- 10s 39ms/step - val_loss: 0.3886 - val_accuracy: 0.8350


#### Plotting experiment history.
``` python
metrics = ["loss" , "accuracy"]
for key in metrics:
    exp.plot_history(key = key , save_fig = False , plot_fig = True)
```



![png](Imdb_classification_files/Imdb_classification_7_0.png)





![png](Imdb_classification_files/Imdb_classification_7_1.png)

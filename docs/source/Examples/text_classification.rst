Text Classfication
====================================


1. Import the necessary libraries

.. code-block:: python

    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    import torch
    import torch.nn as nn

    import transformers
    import torchflare.callbacks as cbs
    import torchflare.metrics as metrics
    import torchflare.criterion as crit
    from torchflare.experiments import Experiment,ModelConfig
    from torchflare.datasets import TextDataloader

2. Read the data and prepare dataloaders

.. code-block:: python

    train_df , valid_df =train_test_split(df , stratify = df.label,  test_size = 0.1, random_state = 42)
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

3. Define the model

.. code-block:: python

    class Model(torch.nn.Module):

        def __init__(self,dropout , out_features):

            super(Model , self).__init__()
            self.bert = transformers.BertModel.from_pretrained(
                "prajjwal1/bert-tiny", return_dict=False
            )
            self.bert_drop = nn.Dropout(dropout)
            self.out = nn.Linear(128, out_features)

        def forward(self, x):
            _ , o_2 = self.bert(**x)

            b_o = self.bert_drop(o_2)
            output = self.out(b_o)
            return output

4. Define model config, callbacks and metrics.

.. code-block:: python


    metric_list = [metrics.Accuracy(num_classes=2, multilabel=False)]

    callbacks = [
        cbs.EarlyStopping(monitor="val_accuracy", patience=2, mode = "max"),
        cbs.ModelCheckpoint(monitor="val_accuracy" , mode = "max", save_dir = "./",
                           file_name = "model.bin"),
        cbs.ReduceLROnPlateau(mode = "max" , patience = 2)
    ]

    config = ModelConfig(nn_module = Model, module_params = {"dropout" : 0.3 , "out_features" : 1}
                         , optimizer = "AdamW",optimizer_params = {"lr" : 3e-4},
                        criterion = crit.BCEWithLogitsFlat)

5. Compile and train the model.

.. code-block:: python

    exp = Experiment(
        num_epochs=3,
        fp16=True,
        device="cuda",
        seed=42,
    )

    # Compiling the experiment
    exp.compile_experiment(
        model_config = config,
        callbacks = callbacks,
        metrics=metric_list,
        main_metric="accuracy",
    )

    # Training the models.
    exp.fit_loader(train_dl = train_dl , valid_dl = valid_dl)

More examples are available in `Github repo <https://github.com/Atharva-Phatak/torchflare/tree/main/examples>`_.

::: torchflare.datasets.text_dataloader.TextDataloader
    handler: python
    selection:
       members:
         - from_df
         - get_loader
    rendering:
      show_root_full_path: false
      show_root_toc_entry: false
      show_root_heading: false
      show_source: false


* ### from_df

``` python

from torchflare.datasets import TextDataloader

dl = TextDataloader.data_from_df(df=df,
                                  input_col="tweet",
                                  label_cols="label",
                                  tokenizer=tokenizer,
                                   max_len=128
                                   ).get_loader(batch_size=64, # Required Args.
                                                       shuffle=True, # Required Args.
                                                       num_workers = 0, # keyword Args.
                                                       collate_fn = collate_fn # keyword Args.
                                                       )

```

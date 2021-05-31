Datasets
===========================
.. toctree::
   :titlesonly:

.. contents::
   :local:

torchflare.datasets.cross\_val module
-------------------------------------

.. automodule:: torchflare.datasets.cross_val
   :members:
   :undoc-members:
   :show-inheritance:

Image Dataset
------------------------------------------------

.. autoclass:: torchflare.datasets.image_classification.ImageDataset
   :members: from_df , from_folders
   :exclude-members: __init__


Segmentation Dataset
---------------------------------------

.. autoclass:: torchflare.datasets.segmentation.SegmentationDataset
   :members: from_rle , from_folders
   :exclude-members: create_mask_list

Tabular Dataset
----------------------------------

.. autoclass:: torchflare.datasets.tabular.TabularDataset
   :members: from_df , from_csv


Text Dataset
----------------------------------------

.. autoclass:: torchflare.datasets.text_dataset.TextClassificationDataset
   :members: from_df

Dataloaders
===========================
.. toctree::
   :titlesonly:

.. contents::
   :local:

Image Dataloader
--------------------------------------------

.. autoclass:: torchflare.datasets.image_dataloader.ImageDataloader
   :members: from_df, from_csv, from_folders, get_loader


Segmentation Dataloader
---------------------------------------------------

.. autoclass:: torchflare.datasets.segmentation_dataloader.SegmentationDataloader
   :members: from_rle , from_folders, get_loader



Tabular Dataloader
----------------------------------------------

.. autoclass:: torchflare.datasets.tabular_dataloader.TabularDataloader
   :members: from_df , from_csv , get_loader

Text Dataloader
-------------------------------------------

.. automodule:: torchflare.datasets.text_dataloader.TextDataloader
   :members: from_df

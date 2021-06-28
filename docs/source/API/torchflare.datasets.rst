Datasets
===========================
.. toctree::
   :titlesonly:

.. contents::
   :local:



Image Dataset
------------------------------------------------

.. autoclass:: torchflare.datasets.ImageDataset
   :members: from_df , from_folders
   :exclude-members: __init__


Segmentation Dataset
---------------------------------------

.. autoclass:: torchflare.datasets.SegmentationDataset
   :members: from_rle , from_folders
   :exclude-members: create_mask_list

Tabular Dataset
----------------------------------

.. autoclass:: torchflare.datasets.TabularDataset
   :members: from_df , from_csv


Text Dataset
----------------------------------------

.. autoclass:: torchflare.datasets.TextClassificationDataset
   :members: from_df

Dataloaders
===========================
.. toctree::
   :titlesonly:

.. contents::
   :local:

Image Dataloader
--------------------------------------------

.. autoclass:: torchflare.datasets.ImageDataloader
   :members: from_df, from_csv, from_folders, get_loader


Segmentation Dataloader
---------------------------------------------------

.. autoclass:: torchflare.datasets.SegmentationDataloader
   :members: from_rle , from_folders, get_loader



Tabular Dataloader
----------------------------------------------

.. autoclass:: torchflare.datasets.TabularDataloader
   :members: from_df , from_csv , get_loader

Text Dataloader
-------------------------------------------

.. autoclass:: torchflare.datasets.TextDataloader
   :members: from_df

Cross Validation
-------------------------------------

.. autoclass:: torchflare.datasets.CVSplit
   :members:

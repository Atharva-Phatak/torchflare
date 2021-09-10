Datasets
===========================

Image Dataset
------------------------------------------------

.. autoclass:: torchflare.datasets.ImageDataset
   :members: from_df , from_folders, from_csv
   :exclude-members: __init__

Tabular Dataset
------------------------------------------------

.. autoclass:: torchflare.datasets.TabularDataset
   :members: from_df , from_csv
   :exclude-members: __init__

Segmentation Dataset
------------------------------------------
.. autoclass:: torchflare.datasets.SegmentationDataset
   :members: from_df , from_folders, masks_from_rle, masks_from_folders, add_test
   :exclude-members: __init__

Tabular Dataset
------------------------------------------------

.. autoclass:: torchflare.datasets.TextDataset
   :members: from_df , from_csv
   :exclude-members: __init__

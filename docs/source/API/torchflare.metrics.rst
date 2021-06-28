Metrics
==========================
.. toctree::
   :titlesonly:

.. contents::
   :local:

Classification Metrics
----------------------------------
Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.Accuracy
   :members:  handle , accumulate , value
   :exclude-members: reset

Precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.precision_meter.Precision
   :members: handle , accumulate , value
   :exclude-members: reset

Recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.recall_meter.Recall
   :members: handle , accumulate , value
   :exclude-members: reset

AUC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.auc.AUC
   :members:


FBeta Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torchflare.metrics.fbeta_meter.FBeta
   :members:  handle , accumulate , value
   :exclude-members: reset

F1 Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.metrics.fbeta_meter.F1Score
   :members:  handle , accumulate , value
   :exclude-members: reset

Segmentation Metrics
-----------------------------

Dice Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.dice_meter.DiceScore
   :members:  handle , accumulate , value
   :exclude-members: reset


IOU Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchflare.metrics.iou_meter.IOU
   :members:  handle , accumulate , value
   :exclude-members: reset

Regression Metrics
--------------------------

MAE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.metrics.regression.MAE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

MSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.metrics.regression.MSE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

MSLE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.metrics.regression.MSLE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

R2Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: torchflare.metrics.regression.R2Score
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

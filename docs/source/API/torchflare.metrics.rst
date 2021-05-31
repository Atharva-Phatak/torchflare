Metrics
==========================
.. toctree::
   :titlesonly:

.. contents::
   :local:


Classification - Accuracy
-----------------------------------------

.. autoclass:: torchflare.metrics.accuracy_meter.Accuracy
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Classification - Precision
------------------------------------------

.. autoclass:: torchflare.metrics.precision_meter.Precision
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Classification - Recall
---------------------------------------

.. autoclass:: torchflare.metrics.recall_meter.Recall
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Classification - AUC
-----------------------------

.. autoclass:: torchflare.metrics.auc.AUC
   :members:
   :undoc-members:
   :show-inheritance:


Classification - FBeta Score
--------------------------------------
.. automodule:: torchflare.metrics.fbeta_meter.FBeta
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Classification - F1 Score
-------------------------
.. autoclass:: torchflare.metrics.fbeta_meter.F1Score
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Segmentation - Dice Score
-------------------------------------

.. autoclass:: torchflare.metrics.dice_meter.DiceScore
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset


Segmentation - IOU Score
------------------------------------

.. autoclass:: torchflare.metrics.iou_meter.IOU
   :members: __init__ , handle , accumulate , value
   :exclude-members: reset

Regression - MAE
-------------------------
.. autoclass:: torchflare.metrics.regression.MAE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

Regression - MSE
-------------------------
.. autoclass:: torchflare.metrics.regression.MSE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

Regression - MSLE
-------------------------
.. autoclass:: torchflare.metrics.regression.MSLE
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

Regression - R2Score
-------------------------
.. autoclass:: torchflare.metrics.regression.R2Score
   :members: accumulate  , value , handle
   :exclude-members: __init__ , reset

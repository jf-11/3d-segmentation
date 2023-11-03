
'''
This file contains the metrics for the model.
'''

# ===============================================================================
# === Imports ===================================================================

import tensorflow as tf
import tensorflow.keras.backend as K

# ===============================================================================
# === MeanIoU metric ============================================================

class MeanIoU(tf.keras.metrics.MeanIoU):
  """
  MeanIoU class which inherits from the tensorflow built in
  MeanIou class.
  """
  def __init__(self,
              y_true=None,
              y_pred=None,
              num_classes=None,
              name=None,
              dtype=None):
    super(MeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)



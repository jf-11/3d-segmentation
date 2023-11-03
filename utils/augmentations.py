
'''
This file contains class for data augmentations.
'''

# ===============================================================================
# === Imports ===================================================================

import tensorflow as tf

# ===============================================================================
# === Data augmentations class ==================================================

class Augment(tf.keras.layers.Layer):
  """
  Class for augmenting the scans.
  """
  def __init__(self, train_config, seed=97):
    self.augmentations = train_config["augmentations"]
    super().__init__()
    self.flip_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.flip_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.rotate_inputs = tf.keras.layers.RandomRotation(factor = 0.2, seed=seed, interpolation='nearest')
    self.rotate_labels = tf.keras.layers.RandomRotation(factor = 0.2, seed=seed, interpolation='nearest')

  def call(self, inputs, labels):
    if "flip" in self.augmentations:
      inputs = self.flip_inputs(inputs)
      labels = self.flip_labels(labels)
    if "rotate" in self.augmentations:
      inputs = self.rotate_inputs(inputs)
      labels = self.rotate_labels(labels)
    return inputs, labels

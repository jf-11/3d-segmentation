
'''
Architechtures for the neural networks build with tensorflow and the keras API.
'''

# ===============================================================================
# === Imports ===================================================================

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation,Dropout,Layer,ReLU,Add,UpSampling2D,Softmax

# ===============================================================================
# === Unet1024 ===================================================================

""" Olaf Ronneberger et al. “U-Net: 
Convolutional Networks for Biomedical Image Segmentation”, 18. May 2015."""

# ===============================================================================

def convolution_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder(input, num_filters):
    x = convolution_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p   

def decoder(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes):
    """
    Function to build the Unet1024 model.
    """
    inputs = Input(input_shape)

    skip1, pool1 = encoder(inputs, 64)
    skip2, pool2 = encoder(pool1, 128)
    skip3, pool3 = encoder(pool2, 256)
    skip4, pool4 = encoder(pool3, 512)

    bridge = convolution_block(pool4, 1024)

    decode1 = decoder(bridge, skip4, 512)
    decode2 = decoder(decode1, skip3, 256)
    decode3 = decoder(decode2, skip2, 128)
    decode4 = decoder(decode3, skip1, 64)

    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax", kernel_regularizer='l2')(decode4)

    model = Model(inputs, outputs, name="U-Net1024")
    return model

# ===============================================================================

# run this file to look at the structure of the model.
if __name__ == "__main__":
    x, y, in_channels, out_channels = 256, 256, 1, 7
    unet = build_unet((x,y,in_channels),out_channels)
    unet.summary()


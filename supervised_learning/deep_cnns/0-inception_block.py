#!/usr/bin/env python3
"""inception block with keras"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """inception block with keras"""
    F1, F3R, F3, F5R, F5, FPP = filters

    layer_1 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    layer_1 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(layer_1)

    layer_2 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    layer_2 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(layer_2)

    layer_3 = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    layer_3 = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(layer_3)

    layer_4 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    mid_1 = K.layers.concatenate([layer_1, layer_2, layer_3, layer_4], axis=3)

    return mid_1

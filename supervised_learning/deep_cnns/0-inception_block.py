#!/usr/bin/env python3
"""inception block with keras"""
from tensorflow import keras as K


def inception_block(X, filters):
    """inception block with keras"""
    F1, F3R, F3, F5R, F5, FPP = filters

    l4 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(X)

    l1 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(X)
    l1 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(l1)

    l2 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(X)
    l2 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(l2)

    l3 = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(X)
    l3 = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(l3)

    mid_1 = K.layers.concatenate([l4, l1, l2, l3], axis=3)

    return mid_1

#!/usr/bin/env python3
"""projection block in keras"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """projection block in keras"""
    F11, F3, F12 = filters
    h_n = K.initializers.HeNormal(seed=0)

    x_shortcut = A_prev

    x = K.layers.Conv2D(F11, (1, 1),
                        strides=s,
                        kernel_initializer=h_n,
                        padding='same')(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F3, (3, 3),
                        kernel_initializer=h_n,
                        padding='same')(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F12, (1, 1),
                        kernel_initializer=h_n,
                        padding='same')(x)

    y = K.layers.Conv2D(F12, (1, 1),
                        strides=s,
                        kernel_initializer=h_n,
                        padding='same')(x_shortcut)
    x = K.layers.BatchNormalization(axis=3)(x)
    y = K.layers.BatchNormalization(axis=3)(y)

    x = K.layers.Add()([x, y])
    x = K.layers.Activation('relu')(x)
    return x

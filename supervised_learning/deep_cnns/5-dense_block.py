#!/usr/bin/env python3
"""dense block in keras"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """dense block in keras"""

    h_n = K.initializers.HeNormal(seed=0)
    x = X
    x_prev = X
    for i in range(layers):
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(growth_rate * 4, (1, 1),
                            padding='same',
                            kernel_initializer=h_n)(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation("relu")(x)
        x = K.layers.Conv2D(32, (3, 3), padding='same',
                            kernel_initializer=h_n)(x)
        x = K.layers.Concatenate()([x_prev, x])
        x_prev = x
    return x, x.shape[3]

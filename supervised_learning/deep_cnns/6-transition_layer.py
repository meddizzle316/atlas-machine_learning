#!/usr/bin/env python3
"""transition layer in keras"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """transition layer in keras"""
    h_n = K.initializers.HeNormal(seed=0)
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(nb_filters * compression, (1, 1),
                        padding='same',
                        kernel_initializer=h_n)(x)
    x = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    return x, x.shape[3]

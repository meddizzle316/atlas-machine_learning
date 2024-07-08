#!/usr/bin/env python3
"""identity block with keras"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """identity block with keras"""
    F11, F3, F12 = filters
    # K.set_random_seed(0) we're supposed to do this, I think?
    # but since Keras 2.3 it's tf.random.set_seed()
    h_n = K.initializers.HeNormal(seed=0)

    X_shortcut = A_prev

    X = K.layers.Conv2D(F11, (1, 1),
                        kernel_initializer=h_n, padding='same')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), kernel_initializer=h_n, padding='same')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), kernel_initializer=h_n, padding='same')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X

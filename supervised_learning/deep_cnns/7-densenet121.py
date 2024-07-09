#!/usr/bin/env python3
"""densenet121 in keras"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """densenet121 in keras"""
    h_n = K.initializers.HeNormal(seed=0)
    i = K.Input(shape=(224, 224, 3))
    x = K.layers.BatchNormalization()(i)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=h_n)(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(x)
    x, y = dense_block(x, 56, growth_rate, 6)
    x, y = transition_layer(x, y, compression)
    x, y = dense_block(x, 28, growth_rate, 12)
    x, y = transition_layer(x, y, compression)
    x, y = dense_block(x, 14, growth_rate, 24)
    x, y = transition_layer(x, y, compression)
    x, y = dense_block(x, 7, growth_rate, 16)
    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        padding='same')(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=h_n)(x)
    return K.models.Model(i, x)

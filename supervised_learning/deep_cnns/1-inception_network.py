#!/usr/bin/env python3
"""inception network with keras"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """inception network with keras"""

    units = None
    i = K.layers.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(64, (7, 7),
                        strides=(2, 2), activation='relu', padding='same')(i)
    # x = K.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2), padding='same')(x)
    # not sure why the following Conv2d layer is there
    # as it doesn't seem to be there in the network
    # and it's not (as far as I can tell)
    # from the inception module output
    # but the 'output' from the main file expects it
    # if problems, check here first
    # seems to have correct output if 1x1 patch size
    x = K.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2), padding='same')(x)
    x = inception_block(x, (64, 96, 128, 16, 32, 32))
    x = inception_block(x, (128, 128, 192, 32, 96, 64))
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2), padding='same')(x)
    x = inception_block(x, (192, 96, 208, 16, 48, 64))
    x = inception_block(x, (160, 112, 224, 24, 64, 64))
    x = inception_block(x, (128, 128, 256, 24, 64, 64))
    x = inception_block(x, (112, 144, 288, 32, 64, 64))
    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2), padding='same')(x)
    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = inception_block(x, (384, 192, 384, 48, 128, 128))
    x = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(1000, activation='softmax')(x)
    return K.models.Model(i, x)

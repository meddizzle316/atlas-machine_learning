#!/usr/bin/env python3
"""lenet 5"""
from tensorflow import keras as K


def lenet5(input):
    """builds lenet 5 in keras"""

    # print(input)
    # print(input.shape)
    m, h, w, c = input.shape

    he_normal = K.initializers.VarianceScaling(scale=2.0)
    x = K.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(m, h, w, c), padding='same', kernel_initializer=he_normal)(input)
    x = K.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = K.layers.Conv2D(16, (5, 5), padding='valid', kernel_initializer=he_normal, activation='relu')(x)
    x = K.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(120, activation='relu', kernel_initializer=he_normal)(x)
    x = K.layers.Dense(84, activation='relu', kernel_initializer=he_normal)(x)
    x = K.layers.Dense(10, activation='softmax', kernel_initializer=he_normal)(x)


    model = K.models.Model(input, x)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

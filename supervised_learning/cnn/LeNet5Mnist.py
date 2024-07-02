#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf1
import tensorflow as tf


lenet5 = __import__('4-lenet5').lenet5

if __name__ == "__main__":

    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)
    tf1.set_random_seed(SEED)
    np.random.seed(SEED)

    lib = np.load('MNIST.npz')
    X_train = lib['X_train']
    Y_train = lib['Y_train']
    X_valid = lib['X_valid']
    Y_valid = lib['Y_valid']
    m, h, w = X_train.shape
    X_train_c = X_train.reshape((-1, h, w))
    X_valid_c = X_valid.reshape((-1, h, w))
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), input_shape=(28, 28, 1), activation='relu', kernel_initializer=he_normal, padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer=he_normal, padding='valid'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu', kernel_initializer=he_normal),
        tf.keras.layers.Dense(84, activation='relu', kernel_initializer=he_normal),
        tf.keras.layers.Dense(10, activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    r = model.fit(X_train, Y_train, epochs=10, steps_per_epoch=200)


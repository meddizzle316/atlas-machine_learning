#!/usr/bin/env python3
"""tensorflow batch norm"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """tensorflow batch normalization"""

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    # base_layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer, activation=activation)
    # input = tf.constant(prev)
    # output = base_layer(input)
    # return tf.keras.layers.BatchNormalization(output)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n, kernel_initializer=initializer, activation=activation),
        tf.keras.layers.BatchNormalization()
    ])
    return model(prev)

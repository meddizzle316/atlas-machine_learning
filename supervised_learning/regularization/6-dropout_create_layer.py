#!/usr/bin/env python3
"""layer with dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """tensorflow dropout layer"""
    drop_out = tf.keras.layers.Dropout(1 - keep_prob)
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    result = tf.keras.layers.Dense(n, activation=activation,
                                   kernel_initializer=init,
                                   kernel_regularizer=drop_out)(prev)
    return result
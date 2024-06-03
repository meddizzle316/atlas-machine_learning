#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def create_layer(prev, n, activation):
    """creates a tensorflow layer"""
    layer = tf.keras.layers.Dense(n,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
                                  activation= 'tanh',
                                  name='layer')

    output= layer(prev)
    return output

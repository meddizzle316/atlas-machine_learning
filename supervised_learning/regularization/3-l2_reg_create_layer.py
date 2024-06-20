#!/usr/bin/env python3
"""l2 reg layer in tf"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """l2 reg layer"""
    l2_reg = tf.keras.regularizers.l2(l2=lambtha)
    scaling = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation, kernel_regularizer=l2_reg, kernel_initializer=scaling)
    return layer(prev)

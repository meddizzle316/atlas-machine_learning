#!/usr/bin/env python3
"""l2 reg layer in tf"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """l2 reg layer"""
    return tf.keras.layers.Dense(n, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(lambtha))(prev)
    
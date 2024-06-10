#!/usr/bin/env python3
"""for tensor adam"""
import tensorflow as tf


def create_Adam_op(a, b1, b2, e):
    """tensorflow adam"""
    return tf.keras.optimizers.Adam(learning_rate=a, beta_1=b1, beta_2=b2, epsilon=e)

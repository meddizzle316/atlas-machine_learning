#!/usr/bin/env python3
"""tensorflow RMSprop"""
import tensorflow as tf


def create_RMSProp_op(alpha, b, e):
    """tensorflow RMSprop"""
    return tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=b, epsilon=e)

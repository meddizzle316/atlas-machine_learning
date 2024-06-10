#!/usr/bin/env python3
"""for upgraded momentum"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """creating momentum op in Tensorflow"""
    return tf.keras.optimizers.SGD(alpha, beta1)

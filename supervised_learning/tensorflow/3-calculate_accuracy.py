#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """predicts accuracy"""
    result = tf.math.divide(y, y_pred)
    named_result = tf.reshape(result, [-1], name='Mean')
    return named_result

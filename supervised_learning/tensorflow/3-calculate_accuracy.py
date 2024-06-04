#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """predicts accuracy"""
    # result = tf.math.divide(y, y_pred)
    # named_result = tf.reshape(result, [-1], name='Mean')
    
    percent_correct = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y, axis=-1))
    
    named_result = tf.reduce_mean(tf.cast(percent_correct, tf.float32))
    return named_result

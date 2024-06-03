#!/usr/bin/env python3
"""stupid tf1 thing even though tf2 is a thing"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def create_placeholders(nx, classes):
    """returns two place holders"""
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y

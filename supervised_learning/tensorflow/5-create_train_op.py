#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def create_train_op(loss, alpha):
    """creates training operation"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op

#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

def calculate_loss(y, y_pred):
    """using softmax_cross_entropy"""

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return loss

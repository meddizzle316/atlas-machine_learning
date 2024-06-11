#!/usr/bin/env python3
"""learning_rate decay tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_steps):
    """learning rate tensorflow"""
    return tf.keras.optimizers.schedules.InverseTimeDecay(alpha, decay_steps, decay_rate, staircase=True
    )

#!/usr/bin/env python3
"""tensorflow RMSprop"""
import tensorflow as tf

def create_RMSProp_op(alpha, beta2, epsilon):
    """tensorflow RMSprop"""
    return tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)

#!/usr/bin/env python3
"""one hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """one hot"""
    return K.utils.to_categorical(labels, classes)

#!/usr/bin/env python3
"""predicts with a model"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predicts with a model"""
    return network.predict(data, verbose=verbose)

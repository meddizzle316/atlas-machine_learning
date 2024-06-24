#!/usr/bin/env python3
"""tests a network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a model"""
    return network.evaluate(data, labels, verbose=verbose)

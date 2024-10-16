#!/usr/bin/env python3
"""saves and loads weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """saves weights"""
    network.save_weights(filename, save_format=save_format)

def load_weights(network, filename):
    """loads weights"""
    network.load_weights(filename)

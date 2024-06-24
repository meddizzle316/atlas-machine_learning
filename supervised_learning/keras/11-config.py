#!/usr/bin/env python3
"""saves and loads config"""
import tensorflow.keras as K
import json


def save_config(network, filename):
    """saves config"""
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)

def load_config(filename):
    """loads config"""
    with open(filename, 'r') as file:
        config_json = file.read()
        model = K.models.model_from_json(config_json)
    return model

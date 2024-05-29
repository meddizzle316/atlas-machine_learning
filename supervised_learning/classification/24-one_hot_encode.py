#!/usr/bin/env python3
"""for onehot_encoding"""
import numpy as np

def one_hot_encode(Y, classes):
    """encoding function"""
    encoded_data = np.zeros((len(Y), classes))

    encoded_data[Y, np.arange(Y.size)] = 1

    return encoded_data

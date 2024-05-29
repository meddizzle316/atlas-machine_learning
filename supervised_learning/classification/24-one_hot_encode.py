#!/usr/bin/env python3
"""for onehot_encoding"""
import numpy as np

def one_hot_encode(Y, classes):
    """encoding function"""
    encoded_data = np.zeros((len(Y), classes))

    for i, val in enumerate(Y):
        encoded_data[i, val] = 1

    return encoded_data

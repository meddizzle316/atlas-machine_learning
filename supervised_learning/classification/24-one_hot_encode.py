#!/usr/bin/env python3
"""for onehot_encoding"""
import numpy as np

def one_hot_encode(Y, classes):
    """encoding function"""
    if type(Y) is not np.ndarray or not isinstance(classes, int):
        return None
    if classes < 2 or classes < np.amax(Y):
        return None
    encoded_data = np.zeros((len(Y), classes))

    encoded_data[Y, np.arange(Y.size)] = 1

    return encoded_data

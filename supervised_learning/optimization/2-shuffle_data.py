#!/usr/bin/env python3
"""module for doing shuffling"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles data points in two matrices the same way"""

    shuffled_X = np.empty(X.shape)
    shuffled_Y = np.empty(Y.shape)
    permutation = np.random.permutation(len(X))
    for old_index, new_index in enumerate(permutation):
        shuffled_X[new_index] = X[old_index]
        shuffled_Y[new_index] = Y[old_index]
    return shuffled_X, shuffled_Y

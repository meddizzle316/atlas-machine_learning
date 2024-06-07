#!/usr/bin/env python3
"""module for doing shuffling"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles data points in two matrices the same way"""
    p = np.random.permutation(len(X))
    # creates a random permutation of numbers 0 up to len(X) - 1 
    # so 4, making a list of 5 numbers 
    # for our particular seed it's [0 2 1 4 3] as the output
    # then we return those indexes of the X and Y matrices
    return X[p], Y[p]

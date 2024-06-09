#!/usr/bin/env python3
"""module for mini batch training from scratch"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """creates mini-batches, doesn't train"""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_index in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_index:start_index + batch_size]
        yield X[excerpt], Y[excerpt]


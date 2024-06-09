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
        # outputs numbers like [11774  3876 28702    66 18965  5006 37642 17812  1403 37740 27519  4449
        #   6649 46204  6537 39997  7660 34661 21377 39712 10973 47200 41241 26995
        #   7543 21504 34858 29257 12941 43542   651  3361]
        yield X[excerpt], Y[excerpt]

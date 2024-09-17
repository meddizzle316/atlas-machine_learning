#!/usr/bin/env python3
"""intra variance of dataset"""
import numpy as np


def variance(X, C):
    """X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means of each cluster
         You are not allowed to use any loops
        Returns: var, or None on failure
        var is the total variance"""

    # Calculate squared distances
    # Assign each data point to its nearest centroid
    distances = np.min(np.linalg.norm(X[:, None] - C, axis=2), axis=1)

    # Calculate the squared distances
    squared_distances = distances ** 2

    # Calculate the total intra-cluster variance
    variance = np.sum(squared_distances)

    return variance


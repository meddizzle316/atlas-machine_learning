#!/usr/bin/env python3
"""intra variance of dataset"""
import numpy as np


def variance(X, C):
    """X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means of each cluster
         You are not allowed to use any loops
        Returns: var, or None on failure
        var is the total variance"""
    try:
        n, d = X.shape
        k, d1 = C.shape
    except Exception:
        return None
    if not X.ndim == 2 or C.ndim == 2:
        return None


    # Calculate squared distances
    distances = np.min(np.linalg.norm(X[:, None] - C, axis=2), axis=1)

    # Calculate the squared distances
    squared_distances = distances ** 2

    # Calculate the total intra-cluster variance
    variance = np.sum(squared_distances)

    return variance


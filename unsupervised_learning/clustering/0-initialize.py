#!/usr/bin/env python3
"""initialize cluster centriods for k-means"""
import numpy as np


def initialize(X, k):
    """initialize cluster centriods for k-means
        X is a numpy.ndarray of shape (n, d) containing the dataset
        that will be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters

        The cluster centroids should be initialized with a multivariate
        uniform distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
        values of X along each dimension in d
        The maximum values for the distribution should be the maximum
        values of X along each dimension in d
        You should use numpy.random.uniform exactly once
        You are not allowed to use any loops

        Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if k <= 0 or not isinstance(k, int):
        return None
    if not isinstance(X, np.ndarray) or X.ndim < 2:
        return None
    m, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    centroids = np.random.uniform(min_val, max_val, size=(k, d))
    return centroids

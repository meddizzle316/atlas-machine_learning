#!/usr/bin/env python3
"""initializes variables for Gassian Mixture model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """X is a numpy.ndarray of shape (n, d)
    containing the data set
    k is a positive integer containing the number of clusters

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing
        the priors for each cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster, initialized
        with K-means
        S is a numpy.ndarray of shape (k, d, d) containing
        the covariance matrices for each cluster, initialized
        as identity matrices"""

    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None, None

    if k <= 0 or X.ndim != 2:
        return None, None, None

    return None, None, None

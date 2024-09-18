#!/usr/bin/env python3
"""gets optimum number of clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
    X:
    A numpy.ndarray of shape (n, d) containing
    the data set.
    kmin: A positive integer containing the minimum
    number of clusters to check for (inclusive).
    kmax: A positive integer containing the maximum
    number of clusters to check for (inclusive).
    iterations: A positive integer containing the
    maximum number of iterations for K-means.

    Returns:
    results: A list containing the outputs of K-means
    for each cluster size.
    d_vars: A list containing the difference in variance
    from the smallest cluster size for each cluster size.
    """

    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if not isinstance(iterations, int) or not isinstance(X, np.ndarray):
        return None, None

    if kmin <= 0 or kmax <= 0:
        return None, None
    if X.ndim != 2:
        return None, None

    return None, None

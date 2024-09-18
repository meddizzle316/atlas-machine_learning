#!/usr/bin/env python3
"""calculates the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """ X is a numpy.ndarray of shape (n, d)
    containing the data set
    g is a numpy.ndarray of shape (k, n) containing
    the posterior probabilities for each data point
    in each cluster"""

    if not isinstance(X, np.ndarray) or isinstance(g, np.ndarray):
        return None, None, None
    if not X.ndim != 2 or not g.ndim != 2:
        return None, None, None

    return None, None, None

#!/usr/bin/env python3
"""creating multinormal class"""
import numpy as np


def mean_cov(X):
    """calculates mean and covariance of given dataset
        Arg: X -- is numpy arry of d, n
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("data must be a 2D numpy.ndarray")
    if not X.ndim >= 2:
        raise TypeError("data must be a 2D numpy.ndarray")
    d, n = X.shape
    if n < 2:
        raise ValueError("data must contain multiple data points")
    mean = np.mean(X, axis=-1)
    resh_mean = np.reshape(mean, (-1, 1))
    cov = np.zeros((d, d))
    X = X.T
    for i in range(d):

        mean_i = np.sum(X[:, i]) / n
        for j in range(d):
            mean_j = np.sum(X[:, j]) / n

            cov[i, j] = np.sum((X[:, i] - mean_i) *
                               (X[:, j] - mean_j)) / (n - 1)

    return resh_mean, cov


class MultiNormal():
    def __init__(self, data):
        """init function for MultNormal class
        sets mean and covariance of given dataset"""
        mean, cov = mean_cov(data)
        self.mean = mean
        self.cov = cov

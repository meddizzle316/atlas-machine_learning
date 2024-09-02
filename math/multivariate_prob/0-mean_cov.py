#!/usr/bin/env python3
"""gets mean and covariance of given dataset"""
import numpy as np


def mean_cov(X):
    """calculates mean and covariance of given dataset"""
    n, d = X.shape

    cov = np.zeros((d, d))
    for i in range(d):

        mean_i = np.sum(X[:, i]) / n
        for j in range(d):
            mean_j = np.sum(X[:, j]) / n

            cov[i, j] = np.sum((X[:, i] - mean_i) * (X[:, j] - mean_j)) / (n - 1)

    mean = np.mean(X, axis=0)
    resh_mean = np.reshape(mean, (1, -1))
    return resh_mean, cov

#!/usr/bin/env python3
import numpy as np


def mean_cov(X):
    """calculates mean and covariance of given datset"""
    return np.mean(X), np.cov(X)

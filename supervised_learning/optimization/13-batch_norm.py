#!/usr/bin/env python3
"""batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """batch norm layer"""
    # Z.shape is (3, 100)
    # m = Z.shape[0] # 100
    # u = np.sum(Z) / m
    # var = np.sum(((Z - u) ** 2)) / m
    # the above approach does calculate the mean
    # and the variance
    # but it does so for the ENTIRE dataset
    u = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    # the above gets mean and variance
    # but for each individual 'feature' 
    # aka by axis=0 the rows
    # which is apparently more appropriate 
    # for batch normalization
    z_norm = (Z - u) / ((var + epsilon) ** 0.5)
    return (z_norm * gamma) + beta

#!/usr/bin/env python3
"""module for doing norm constants"""
import numpy as np


def normalization_constants(X):
    """calculates norm constants"""
    return np.mean(X, axis=0), np.std(X, axis=0)

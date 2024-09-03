#!/usr/bin/env python3
"""gets correlation matrix from given numpy array"""
import numpy as np


def correlation(C):
    """gets correclation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if not C.ndim >= 2:
        raise ValueError("C must be a 2D square matrix")
    d1, d2 = C.shape
    if not d1 == d2:
        raise ValueError("C must be a 2D square matrix")
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlate = C / outer_v
    correlate[C == 0] = 0
    return correlate

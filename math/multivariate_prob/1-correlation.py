#!/usr/bin/env python3
"""gets correlation matrix from given numpy array"""
import numpy as np


def correlation(C):
    """gets correclation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    d1, d2 = C.shape
    if not d1 == d2:
        raise TypeError("C must be a 2D numpy.ndarray")
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlate = C / outer_v
    correlate[C == 0] = 0
    return correlate

#!/usr/bin/env python3
"""gets correlation matrix from given numpy array"""
import numpy as np


def correlation(C):
    """gets correclation matrix"""
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlate = C / outer_v
    correlate[C == 0] = 0
    return correlate

#!/usr/bin/env python3
"""checks for absorbing states"""
import numpy as np


def absorbing(p):
    """checks for absorbing states"""

    # if not isinstance(p, np.ndarray):
    #     return None
    n, m = p.shape
    # if not p.shape == (n, n):
    #     return None
    if np.any(p == 1):
        return True
    elif np.linalg.det(p) == 0:
        return False
    else:
        return False

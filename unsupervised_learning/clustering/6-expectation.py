#!/usr/bin/env python3
"""gets expectation step in EM
for GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """gets expectation step in
    EM for GMM"""
    if not isinstance(
        X, np.ndarray) or not isinstance(
        pi, np.ndarray) or not isinstance(
            m, np.ndarray) or not isinstance(
                S, np.ndarray):
        return None, None
    if (not X.ndim != 2 or not pi.ndim != 1 or not m.ndim != 2
            or not S.ndim != 3):
        return None, None

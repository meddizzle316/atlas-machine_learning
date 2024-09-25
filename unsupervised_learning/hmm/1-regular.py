#!/usr/bin/env python3
"""gets stead state probability of a regular markov chain"""
import numpy as np


def regular(p):
    """determines steady state probability"""
    if not isinstance(p, np.ndarray):
        return None
    n, m = p.shape
    if not p.ndim == 2:
        return None
    if not p.shape == (n, n):
        return None
    row_sums = np.sum(p, axis=1)

    if np.any(row_sums > 1):
        return None
    if np.any(p == 1):
        return None

    dim = p.shape[0]
    q = (p - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    if np.linalg.det(QTQ) == 0:
        return None
    bQT = np.ones(dim)
    result = np.linalg.solve(QTQ, bQT)
    result = result.reshape(1, -1)
    return result

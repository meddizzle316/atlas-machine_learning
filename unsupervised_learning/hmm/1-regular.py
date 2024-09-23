#!/usr/bin/env python3
"""gets stead state probability of a regular markov chain"""
import numpy as np

def regular(p):
    """determines steady state probability"""
    dim = p.shape[0]
    q = (p - np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ, bQT)
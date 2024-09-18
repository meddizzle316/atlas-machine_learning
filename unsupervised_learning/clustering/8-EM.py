#!/usr/bin/env python3
"""does expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """does expectation maximization for a GMM"""
    if not isinstance(
            X,
            np.ndarray) or not isinstance(
            k,
            int) or not isinstance(
                iterations,
                int) or not isinstance(
                    tol,
            float):
        return None, None, None, None, None
    if not X.ndim != 2 or k <= 0 or iterations <= 0 or tol < 0:
        return None, None, None, None, None

    return None, None, None, None, None

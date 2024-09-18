#!/usr/bin/env python3
""" finds the best number of clusters for
a GMM using the Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds the best number of clusters for
    a GMM using the Bayesian Information Criterion"""

    if not isinstance(
            X,
            np.ndarray) or not isinstance(
            kmin,
            int) or not isinstance(
                kmax,
                int) or not isinstance(
                    iterations,
                    int) or not isinstance(
                        tol,
            float):
        return None, None, None, None

    if not X.ndim != 2 or kmin <= 0 or kmax <= 0 or iterations <= 0 or tol < 0:
        return None, None, None, None

    return None, None, None, None

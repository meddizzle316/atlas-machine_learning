#!/usr/bin/env python3
"""
calculates likelihood using binomial pmf
has pmf and baynesian likelihood functions
baynesian depends on binomial pmf function
"""
import numpy as np


def pmf(k, n, p):
    """probability mass function of binomial
    K: corresponds to x in likelihood function
    n: corresponds to n
    P: corresponds to P
    """
    if k < 0:
        return 0
    k = int(k)
    first_factor = np.math.factorial(
        n) / (np.math.factorial(n - k) * (np.math.factorial(k)))
    second_factor = (p ** k) * ((1 - p) ** (n - k))
    return first_factor * second_factor


def likelihood(x, n, P):
    """ gets likelihood of x for n trials with P probability (as threshold)
    ARgs
    X: the given number
    n: number of trials
    P: the given probability in numpy format
    """
    if not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if not P.ndim == 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((0 <= P) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    scratch_likeli = []
    for p in P:
        prob = pmf(x, n, p)
        scratch_likeli.append(prob)
    np_array = np.array(scratch_likeli)
    return np_array

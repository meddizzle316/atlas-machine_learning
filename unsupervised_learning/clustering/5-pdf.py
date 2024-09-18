#!/usr/bin/env python3
"""pdf of gassian distribution"""
import numpy as np


def pdf(x, m, s):
    """pdf of gassian distribution"""

    if not isinstance(
        x, np.ndarray) or not isinstance(
        m, np.ndarray) or not isinstance(
            s, np.ndarray):
        return None
    if not x.ndim != 2 or not m.ndim != 1 or not s.ndim != 2:
        return None
    e = 2.7182818285
    pi = 3.1415926536

    # mean = np.mean(x)
    mean = m
    stddev = np.std(x)

    """gets pdf for given x"""
    first_factor = 1 / (((stddev ** 2) * (2 * pi)) ** 0.5)
    second_factor = e ** ((-(x - mean) ** 2) / ((2 * (stddev ** 2))))
    return (first_factor * second_factor)

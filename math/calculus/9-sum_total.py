#!/usr/bin/env python3
import numpy as np
"""does summation notation operations"""


def summation_i_squared(n):
    """squares and adds"""
    if not isinstance(n, int):
        return None
    list = np.linspace(1, n, n)
    return (int(sum(list ** 2)))

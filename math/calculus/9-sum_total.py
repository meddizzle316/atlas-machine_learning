#!/usr/bin/env python3
"""does summation notation operations"""
import numpy as np


def summation_i_squared(n):
    """squares and adds"""
    try:
        test_n = int(n)
        if n == 0:
            return None
    except Exception:
        return None
    list = np.linspace(1, n, n)
    return (int(sum(list ** 2)))

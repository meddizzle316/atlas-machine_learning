#!/usr/bin/env python3
"""does summation notation operations"""


def summation_i_squared(n):
    """squares and adds"""
    if n == 1:
        return 1
    elif isinstance(n, int):
        return (n ** 2) + summation_i_squared(n-1)
    else:
        return None

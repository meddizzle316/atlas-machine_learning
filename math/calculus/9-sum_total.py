#!/usr/bin/env python3
"""does summation notation operations"""


def summation_i_squared(n):
    """squares and adds"""
    return sum(n ** 2 for n in range(1, n + 1))

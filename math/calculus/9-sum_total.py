#!/usr/bin/env python3
"""does summation notation operations"""


def summation_i_squared(n):
    """squares and adds"""
    result = 0
    while n > 0:
        result += (n ** 2)
        n -= 1
    return result

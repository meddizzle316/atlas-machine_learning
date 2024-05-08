#!/usr/bin/env python3
"""does summation notation operations"""
import numpy as np


def summation_i_squared(n):
    """squares and adds couldn't get n=not valid check"""
    list = np.linspace(1, n, n)
    return (int(sum(list ** 2)))

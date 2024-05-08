#!/usr/bin/env python3
"""does summation notation operations"""
import numpy as np
import math



def summation_i_squared(n):
    """squares and adds"""
    if math.isnan(n):
        return None
    list = np.linspace(1, n, n)
    return (int(sum(list ** 2)))

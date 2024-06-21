#!/usr/bin/env python3
"""a module for l2 regularization cost"""
"""a test to see if things work"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """l2 reg cost function"""
    w2 = 0
    for i in range(1, L + 1):
        weight_layer = weights[f"W{i}"]
        w2 += np.sum(np.square(weight_layer))
    second_part = (lambtha / (2 * m)) * (w2)
    return cost + second_part

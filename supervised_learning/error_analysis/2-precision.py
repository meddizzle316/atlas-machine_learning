#!/usr/bin/env python3
"""precision module"""
import numpy as np


def precision(confusion):
    """gets precision of confusion"""
    sum = np.sum(confusion, axis=0)
    true_positive = np.diag(confusion)
    false_positive = sum - true_positive
    return true_positive / (false_positive + true_positive)

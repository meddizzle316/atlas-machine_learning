#!/usr/bin/env python3
"""f1 scoring"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """gets f1 score"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return (np.multiply(np.multiply(2, prec), sens)) / (prec + sens)

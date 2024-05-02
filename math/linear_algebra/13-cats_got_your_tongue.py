#!/usr/bin/env python3
"""function that concantenates using np.concantenate"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function that concantenates using np.concantenate"""
    return np.concatenate((mat1, mat2), axis=axis)

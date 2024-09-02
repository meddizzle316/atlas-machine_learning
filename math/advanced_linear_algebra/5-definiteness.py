#!/usr/bin/env python3
import numpy as np
"""gets definiteness of matrix"""


def definiteness(matrix):
    """gets definitiness of matrix"""
    # check if numpy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # check if empty
    if matrix.size == 0:
        return None

    # check if square/symmetric ??
    for row in matrix:
        if len(row) != len(matrix):
            return None

    eigenvalues = np.linalg.eigh(matrix)[0]

    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None

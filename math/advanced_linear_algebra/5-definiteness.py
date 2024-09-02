#!/usr/bin/env python3
import numpy as np
"""gets definiteness of matrix 
using numpy"""


def definiteness(matrix):
    """gets definitiness of matrix
    using numpy"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    try:
        if not np.allclose(matrix, matrix.T):
            return None
    except ValueError:
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

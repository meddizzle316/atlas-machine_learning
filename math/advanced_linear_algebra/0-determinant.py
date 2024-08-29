#!/usr/bin/env python3
"""doing determinants of matrices"""


def getMatrixMinor(m, i, j):
    """gets minor of given element of matrix"""
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def determinant(matrix):
    """gets determinant of given matrix"""

    try:
        if len(matrix) == 0:
            raise TypeError("matrix must be a list of lists")
        for element in matrix:
            if not isinstance(element, list):
                raise TypeError("matrix must be a list of lists")
    except Exception:
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        return 1
    elif len(matrix[0]) == 1:
        return matrix[0][0]
    elif len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    deter = 0
    for c in range(len(matrix)):
        deter += ((-1)**c) * matrix[0][c] * \
            determinant(getMatrixMinor(matrix, 0, c))
    return deter

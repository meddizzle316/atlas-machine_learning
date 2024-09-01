#!/usr/bin/env python3
"""doing determinants of matrices"""


def getMatrixMinor(m, i, j):
    """gets minor of given element of matrix"""
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def determinant(matrix):
    """gets determinant of given matrix"""

    if matrix == [[]]:
        return 1
    if (
        matrix and matrix[0] and type(matrix) is list
        and all(type(row) is list for row in matrix)
    ):
        # checking if matrix is square
        for row in matrix:
            if len(matrix) != len(row):
                raise ValueError("matrix must be a square matrix")

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    deter = 0
    for c in range(len(matrix)):
        deter += ((-1)**c) * matrix[0][c] * \
            determinant(getMatrixMinor(matrix, 0, c))
    return deter

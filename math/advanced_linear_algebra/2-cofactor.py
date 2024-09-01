#!/usr/bin/env python3
""""gets cofactor"""


#!/usr/bin/env python3
"""gets minor of given matrix"""


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
        if len(matrix) == 1:
            return matrix[0][0]
    else:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    deter = 0
    for c in range(len(matrix)):
        deter += ((-1)**c) * matrix[0][c] * \
            determinant(getMatrixMinor(matrix, 0, c))
    return deter


def getMatrixMinor(m, i, j):
    """gets minor of given element of matrix"""
    # return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]
    minor = []
    for row in (m[:i] + m[i + 1:]):
        minor.append(row[:j] + row[j + 1:])
    print(sum(minor[0]))
    return sum(minor[0])


def minor(matrix):
    """gets minor matrix of given matrix"""
    # minor_m = zeros_like(len(matrix))
    try:
        if len(matrix) == 0:
            raise TypeError("matrix must be a list of lists")
        for element in matrix:
            if not isinstance(element, list):
                raise TypeError("matrix must be a list of lists")
    except Exception:
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        return [1]
    elif len(matrix[0]) == 1:
        return matrix[0]

    # checking if matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    minor_matrix = []
    for rows in range(len(matrix)):
        row_matrix = []
        for column in range(len(matrix[0])):
            # iterating through each element of the matrix

            # getting submatrix
            submatrix = [[x for col, x in enumerate(
                row) if col != column and i != rows]
                         for i, row in enumerate(matrix)]

            # removing empty elements
            result = []
            for sub in submatrix:
                if sub:
                    result.append(sub)

            # getting determinant of submatrix
            element_minor = determinant(result)
            row_matrix.append(element_minor)
        minor_matrix.append(row_matrix)

    return minor_matrix


def cofactor(matrix):
    """gets cofactor"""
    if matrix == [[]]:
        return 1
    if (
        matrix and matrix[0] and type(matrix) is list
        and all(type(row) is list for row in matrix)
    ):
        # checking if matrix is square
        for row in matrix:
            if len(matrix) != len(row):
                raise ValueError("matrix must be a non-empty square matrix")
        # if len(matrix) == 1:
        #     return matrix[0][0]
    else:
        raise TypeError("matrix must be a list of lists")
    cofactors = []
    minor_matrix = minor(matrix)

    for row in range(len(matrix)):
        row_minor = []
        for column in range(len(matrix)):
            row_minor.append(((-1) **(row + column)) * minor_matrix[row][column] )
        cofactors.append(row_minor)

    return cofactors
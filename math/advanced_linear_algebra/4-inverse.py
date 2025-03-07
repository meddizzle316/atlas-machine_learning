#!/usr/bin/env python3
"""gets the adjugate of a matrix"""


def getMatrixDeternminant(m):
    """# base case for 2x2 matrix"""
    if len(m) == 1:
        return m[0][0]

    if len(m) == 2:
        return (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * 1.0

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1.0)**c) * \
            m[0][c] * getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def determinant(matrix):
    """gets determinant of given matrix"""

    if len(matrix) == 1:
        return matrix[0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    deter = 0
    for c in range(len(matrix)):
        deter += ((-1)**c) * matrix[0][c] * \
            determinant(getMatrixMinor(matrix, 0, c))
    return deter


def getMatrixMinor(m, i, j):
    """gets minor of given element of matrix"""
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


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
        return [[1]]
    elif len(matrix[0]) == 1:
        return [[1]]

    # checking if matrix is square
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError("matrix must be a non-empty square matrix")

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
            row_minor.append(((-1) ** (row + column)) *
                             minor_matrix[row][column])
        cofactors.append(row_minor)

    return cofactors


def transpose(matrix):
    """transposes a matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def adjugate(matrix):
    """gets adjugate"""

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
            if len(matrix) == 1 and len(row) == 1:
                # return matrix[0][0]
                return [[1]]
    else:
        raise TypeError("matrix must be a list of lists")

    co_matrix = cofactor(matrix)
    adj_matrix = transpose(co_matrix)

    return adj_matrix


def inverse(matrix):
    """gets inverse of matrix"""
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
            # if len(matrix) == 1 and len(row) == 1:
            #     # return matrix[0][0]
            #     return [[1]]
    else:
        raise TypeError("matrix must be a list of lists")

    deter = getMatrixDeternminant(matrix)
    if deter == 0:
        return None
    if len(matrix) == 2:
        return [[matrix[1][1] / deter, -1 * matrix[0][1] / deter],
                [-1 * matrix[1][0] / deter, matrix[0][0] / deter]]

    cofactors = []
    for r in range(len(matrix)):
        cofactorRow = []
        for c in range(len(matrix)):
            minor = getMatrixMinor(matrix, r, c)
            cofactorRow.append(((-1)**(r + c) * determinant(minor)))
        cofactors.append(cofactorRow)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / deter
    return cofactors

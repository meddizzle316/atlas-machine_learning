#!/usr/bin/env python3
"""function that adds two matrices element wise"""


def matrix_dimensions(matrix):
    """function that finds the dimensions of a given matrix"""
    if isinstance(matrix, list):
        if matrix and isinstance(matrix[0], list):
            return 1 + matrix_dimensions(matrix[0])
        else:
            return 1
    else:
        return 0


def matrix_length(matrix, x, n):
    """function that finds the length of matrix at dimension n"""
    if isinstance(matrix, list):
        if matrix and isinstance(matrix[0], list) and x < n:
            x += 1
            return matrix_length(matrix[0], x, n)
        elif matrix and x == n:
            return len(matrix)
    else:
        return 0


def matrix_shape(matrix):
    """function that returns a list of the shape of the matrix"""
    result = []
    number_of_dimensions = matrix_dimensions(matrix)
    i: int = 0
    while i < number_of_dimensions:
        try:
            result.append(matrix_length(matrix, 0, i))
        except TypeError:
            pass
        i += 1
    return result


def add_matrices2D(mat1, mat2):
    """function that adds 2 2d matrices together"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(mat1[0]) == 0:
        return []
    i: int = 0
    new_matrix = []
    while i < matrix_shape(mat1)[0]:
        list = []
        x = 0
        while x < matrix_shape(mat1)[1]:
            list.append(mat1[i][x] + mat2[i][x])
            x += 1
        new_matrix.append(list)
        i += 1
    return new_matrix

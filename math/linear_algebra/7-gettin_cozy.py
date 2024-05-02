#!/usr/bin/env python3
"""module that concantenates two arrays along a single axis"""


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


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two 2d matrices with optional axis"""
    new_matrix = []

    if mat1 == [[]] or mat2 == [[]]:
        return None
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if (len(mat1[0]) == 0) or (len(mat2[0]) == 0):
        return None
    if mat1 == [] or mat2 == []:
        return None
    if not mat1:
        return None
    if not mat1[0] or not mat2[0]:
        return None
    if axis == 0:
        new_matrix = mat1 + mat2
    elif axis == 1:
        i: int = 0
        while i < 999:
            list = []
            try:
                list = mat1[i] + mat2[i]
                new_matrix.append(list)
            except IndexError:
                break
            i += 1
    return new_matrix

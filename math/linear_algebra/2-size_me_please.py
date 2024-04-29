#!/usr/bin/env python3
"""module doing what numpy.ndarray does but from scratch """


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

    # result = []
    # first: int = 0
    # second: int = 0
    # third: int = 0
    # for i, dimension in enumerate(matrix):
    #     first += 1
    #     try:
    #         for x, array in enumerate(dimension):
    #             if x > second:
    #                 second = x
    #             try:
    #                 for y, list in enumerate(array):
    #                     if y > third:
    #                         third = y
    #             except TypeError:
    #                 pass
    #     except TypeError:
    #         pass

    # result.append(first)
    # if second > 0:
    #     result.append(second + 1)
    # if third > 0:
    #     result.append(third + 1)
    # return result

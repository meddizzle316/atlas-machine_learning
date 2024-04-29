#!/usr/bin/env python3
import numpy as np
# def matrix_dimensions(matrix):
#     if isinstance(matrix, list):
#         if matrix and isinstance(matrix[0], list):
#             return 1 + matrix_dimensions(matrix[0])
#         else:
#             return 1
#     else:
#         return 0

# def matrix_shape(matrix):
#     result = []
#     number_of_dimensions = matrix_dimensions(matrix)
#     for i in range(number_of_dimensions):
#         matrix


def matrix_shape(matrix):
    numpy_matrix = np.array(matrix)
    return numpy_matrix.shape














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



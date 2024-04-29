#!/usr/bin/env python3

def matrix_dimensions(matrix):
    if isinstance(matrix, list):
        if matrix and isinstance(matrix[0], list):
            return 1 + matrix_dimensions(matrix[0])
        else:
            return 1
    else:
        return 0
    
def matrix_length(matrix, x, n):
    if isinstance(matrix, list):
        if matrix and isinstance(matrix[0], list) and x < n:
            x += 1
            return matrix_length(matrix[0], x, n)
        elif matrix and x == n: 
            return len(matrix)
    else:
        return 0

def matrix_shape(matrix):
    result = []
    number_of_dimensions = matrix_dimensions(matrix)
    # print(number_of_dimensions)
    i: int = 0
    while i < number_of_dimensions:
        try:
            result.append(matrix_length(matrix, 0, i))
        except TypeError:
            pass
        i += 1
    # length_of_first = matrix_length(matrix, 0, 0)
    # print(length_of_first)
    # length_of_second = matrix_length(matrix, 0, 1)
    # print(length_of_second)
    # length_of_third = matrix_length(matrix, 0, 2)
    # print(length_of_third)
    # for i in matrix:
    #     result.append(len(matrix))
    return result


# def matrix_shape(matrix):
#     numpy_matrix = np.array(matrix)
#     return numpy_matrix.shape














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



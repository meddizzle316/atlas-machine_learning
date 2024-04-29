#!/usr/bin/env python3
def matrix_shape(matrix):

    result = []
    first: int = 0
    second: int = 0
    third: int = 0
    for i, dimension in enumerate(matrix):
        first += 1
        try:
            for x, array in enumerate(dimension):
                if x > second:
                    second = x
                try:
                    for y, list in enumerate(array):
                        if y > third:
                            third = y
                except TypeError:
                    pass
        except TypeError:
            pass

    result.append(first)
    if second > 0:
        result.append(second + 1)
    if third > 0:
        result.append(third + 1)
    return result

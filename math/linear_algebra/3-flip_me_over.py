#!/usr/bin/env python3
"""module to manually transpose a 2d matrix"""


def matrix_transpose(matrix):
    """function to manually transpose a 2d matrix"""
    new_matrix = []
    nRow = len(matrix)
    nElements = len(matrix[0])
    i = 0
    while i < nElements:
        list = []
        x = 0
        while x < nRow:
            list.append(matrix[x][i])
            x += 1
        new_matrix.append(list)
        i += 1
    return new_matrix

#!/usr/bin/env python3
"""multiplying matrices"""


def mat_mul(mat1, mat2):
    """function to multiply matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    new_matrix = []
    for i in range(len(mat1)):
        list = []
        for x in range(len(mat2[0])):
            temp = 0
            for k in range(len(mat2)):
                temp += mat1[i][k] * mat2[k][x]
            list.append(temp)
        new_matrix.append(list)
    return new_matrix

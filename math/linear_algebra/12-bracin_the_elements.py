#!/usr/bin/env python3
"""module that performs element wise basic operations"""


def np_elementwise(mat1, mat2):
    """function that does elementwise operations"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

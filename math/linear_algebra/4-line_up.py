#!/usr/bin/env python3
"""function that adds 1d arrays of same length"""


def add_arrays(arr1, arr2):
    """function that adds 1d arrays of same length"""
    if len(arr1) != len(arr2):
        return None
    new_list = []
    i = 0
    while i < len(arr1):
        new_list.append(arr1[i] + arr2[i])
        i += 1
    return new_list

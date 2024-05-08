#!/usr/bin/env python3
"""gets the derivative of given polynomial"""


def poly_derivative(poly): 
    """gets derivative of given poly"""
    new_list = []
    i = 1
    while i < len(poly):
        new_list.append(poly[i] * i)
        i += 1
    return new_list

#!/usr/bin/env python3
"""gets the derivative of given polynomial"""


def poly_derivative(poly):
    """gets derivative of given poly"""
    new_list = []
    i = 1
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    all_int = all(isinstance(item, (int)) for item in poly)
    if len(poly) < 1 or not all_int:
        return None

    while i < len(poly):
        new_list.append(poly[i] * i)
        i += 1
    return new_list

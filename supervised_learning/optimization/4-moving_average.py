#!/usr/bin/env python3
"""moving average calculation"""
import numpy as np


def moving_average(data, beta):
    """calculate moving average"""
    """data is the list of data to calculate average from"""
    """beta is the weight used"""

    moving_average = []
    for i in range(len(data)):
        # print(f"data {data[i]}")
        if i > 0:
            # print(f"moving_average[i-1]: {moving_average[i-1]}")
            v = (beta*moving_average[i-1]) + ((1 - beta) * data[i])
        else:
            v = (beta*0) + ((1 - beta) * data[i])
            v = v / (1 - np.power(beta, i + 1))
        # print(f"this is v {v} for index {i}")
        moving_average.append(v)
    # moving_average.pop(0)
    return moving_average

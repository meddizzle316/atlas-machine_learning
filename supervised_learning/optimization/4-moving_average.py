#!/usr/bin/env python3
"""moving average calculation"""
import numpy as np


def moving_average(data, beta):
    """calculate moving average"""
    """data is the list of data to calculate average from"""
    """beta is the weight used"""

    moving_average = [data[0]]
    for i in range(1, len(data)):
        v = (beta*moving_average[i-1]) + ((1 - beta) * data[i])
        moving_average.append(v)
    # moving_average.pop()
    return moving_average

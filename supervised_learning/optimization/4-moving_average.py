#!/usr/bin/env python3
"""moving average calculation"""
import numpy as np


def moving_average(data, beta):
    """calculate moving average"""
    """data is the list of data to calculate average from"""
    """beta is the weight used"""

    moving_average = []
    for i in range(len(data)):
        if i == 0:
            v = (beta*0) + ((1 - beta) * data[i])
        else:
            v = (beta*v) + ((1 - beta) * data[i])
            # this is important: the v above is v calculated 
            # WITHOUT the bias, the bias is only tacked on 
            # just before it's added to the list
            # and we don't touch that list
            # we just use the v term from the last iteration
        bias_v = v / (1 - (beta ** (i + 1)))
        moving_average.append(bias_v)
    # moving_average.pop(0)
    return moving_average

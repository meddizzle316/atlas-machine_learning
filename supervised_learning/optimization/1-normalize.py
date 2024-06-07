#!/usr/bin/env python3
"""module for doing norm constants"""
import numpy as np


def normalize(X, m, s): 
    """normalizes a matrix"""
    return ((X - m) / s)

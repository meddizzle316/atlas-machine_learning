#!/usr/bin/env python3
"""class for neuron"""
import numpy as np


class Neuron:
    """class for neuron"""
    def __init__(self, nx):
        """init"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(np.zeros((1, 784)))
        self.b = 0
        self.A = 0

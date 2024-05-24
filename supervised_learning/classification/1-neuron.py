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
        self.__W = np.random.normal(np.zeros((1, nx)))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b
        
    @property
    def A(self):
        return self.__A

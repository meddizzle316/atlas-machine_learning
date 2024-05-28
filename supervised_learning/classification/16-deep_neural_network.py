#!/usr/bin/env python3
"""defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork():
    """class for deepneuralnetwork"""
    def __init__(self, nx, lay):
        """init func"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(lay, list):
            raise TypeError("layers must be a list of positive integers")

        test_array = np.array([lay])
        if not np.all(test_array > 0) or len(lay) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(lay)
        self.cache = {}
        self.weights = {}

        # setting dimensions and initialization of first layer
        self.weights[f"W{1}"] = np.random.randn(lay[0], nx) * np.sqrt(2/nx)
        self.weights[f"b{1}"] = np.zeros((lay[0], 1))

        # setting dimensions and initialization of 2nd to the last lay
        for i in range(1, self.L):
            # initialize using He et al method (mostly the sqrt at the end?)
            # X: dumb pycode format fix line 35
            X = np.sqrt(2./lay[i - 1])
            self.weights[f"W{i + 1}"] = np.random.randn(lay[i], lay[i - 1]) * X
            self.weights[f"b{i + 1}"] = np.zeros((lay[i], 1))

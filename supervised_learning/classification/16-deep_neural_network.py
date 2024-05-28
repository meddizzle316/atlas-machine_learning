#!/usr/bin/env python3
"""defines a deep neural network"""
import numpy as np
import matplotlib.pyplot as plt

class DeepNeuralNetwork():
    """class for deepneuralnetwork"""
    def __init__(self, nx, layers):
        """init func"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        
        test_array = np.array([layers])
        if not np.all(test_array > 0):    
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        # setting dimensions and initialization of first layer
        self.weights[f"W{1}"] = np.random.randn(layers[0], nx) * np.sqrt(2/nx)
        self.weights[f"b{1}"] = np.zeros((layers[0], 1))

        # setting dimensions and initialization of 2nd to the last layers
        for index in range(1, self.L):
            # initialize using He et al method (mostly the sqrt at the end?)
            self.weights[f"W{index + 1}"] = np.random.randn(layers[index], layers[index-1]) * np.sqrt(2./layers[index -1])
            self.weights[f"b{index + 1}"] = np.zeros((layers[index], 1))

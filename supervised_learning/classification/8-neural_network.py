#!/usr/bin/env python3
"""defines a Neural Network class"""
import numpy as np


class NeuralNetwork:
    """class for Neural Network"""

    def __init__(self, nx, nodes):
        """init"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # weights vector - hidden layer. Random initialization
        self.W1 = np.random.normal(np.zeros((nodes, nx)))

        # bias -- hidden layer. initial value = 0
        self.b1 = 0

        # activated output - hidden layer. initial value = 0
        self.A1 = 0

        # weights vector - output neuron. Random initialization
        self.W2 = np.random.normal(np.zeros((1, nodes)))

        # bais -- output neuron. initial value = 0
        self.b2 = 0

        # Activated output -- output neuron (prediction). initial = 0
        self.A2 = 0

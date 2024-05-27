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
        self.__W1 = np.random.normal(np.zeros((nodes, nx)))

        # bias -- hidden layer. initial value = 0
        self.__b1 = np.zeros((nodes, 1))

        # activated output - hidden layer. initial value = 0
        self.__A1 = 0

        # weights vector - output neuron. Random initialization
        self.__W2 = np.random.normal(np.zeros((1, nodes)))

        # bais -- output neuron. initial value = 0
        self.__b2 = 0

        # Activated output -- output neuron (prediction). initial = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2

    def forward_prop(self, X):
        """forward propogation for neural network"""
        self.__A1 = self.activation(np.dot(self.W1, X) + self.b1)
        self.__A2 = self.activation(np.dot(self.W2, self.A1) + self.b2)

        return (self.__A1, self.__A2)

    def activation(self, x):
        """sigmoid activation function"""
        return 1 / (1 + (np.exp(-x)))

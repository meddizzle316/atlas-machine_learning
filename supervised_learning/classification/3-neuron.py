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
        """it's not clear from the Task that nx should"""
        """be the number of W"""
        self.__W = np.random.normal(np.zeros((1, nx)))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """the weight vectors for the neuron"""
        return self.__W

    @property
    def b(self):
        """bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """the activated output of the neuron(prediction)"""
        return self.__A

    def forward_prop(self, X):
        """func for forward prop"""
        """updates __A"""
        self.__A = self.activation(np.dot(self.W, X) + self.__b)
        # In the model I looked at, the "bias" was just a number
        # added to the input layer
        # but I guess the bias is, in this model
        # included all layers? 
        return self.__A

    def activation(self, x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def cost(self, Y, A):
        """logistic regression binary cost function"""
        """Y is target or correct values"""
        """A is real values or activated output of the neuron"""
        return np.mean(-(np.mean(Y * np.log(A)) + (1.000000001 - Y) * np.log(1.000000001 - A)))
        # the life of me, I couldn't find the -np.mean bit. I guess the -1/m means -mean?

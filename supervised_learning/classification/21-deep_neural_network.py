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

        self.__L = len(lay)
        self.__cache = {}
        self.__weights = {}

        # setting dimensions and initialization of first layer
        self.__weights[f"W{1}"] = np.random.randn(lay[0], nx) * np.sqrt(2 / nx)
        self.__weights[f"b{1}"] = np.zeros((lay[0], 1))

        # setting dimensions and initialization of 2nd to the last lay
        for i in range(1, self.L):
            # initialize using He et al method (mostly the sqrt at the end?)
            # X and rand: dumb pycode format fix line 35
            X = np.sqrt(2. / lay[i - 1])
            rand = np.random.randn(lay[i], lay[i - 1])
            self.__weights[f"W{i + 1}"] = rand * X
            self.__weights[f"b{i + 1}"] = np.zeros((lay[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
    
    def activation(self, x):
        """sigmoid activation function"""
        return 1 / (1 + (np.exp(-x)))

    def forward_prop(self, X):
        """forward propr"""
        
        self.__cache[f"A{0}"] = X
        self.__cache[f"A{1}"] = self.activation(np.dot(self.weights[f'W{1}'], X) + self.weights[f'b{1}'])
        for i in range(2, self.L + 1):
            self.__cache[f"A{i}"] = self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}'])
        return (self.__cache[f"A{self.L}"], self.__cache)

    def cost(self, Y, A):
        """cost function"""
        firstPart = Y * np.log(A)
        secondPart = ((1 - Y) * np.log(1.0000001 - A))
        return np.mean(-(np.mean(firstPart) + (secondPart)))

    def evaluate(self, X, Y):
        """evaluate function"""
        output, cache = self.forward_prop(X)
        mod_output = np.where(output >= 0.5, 1, 0)
        return (mod_output, self.cost(Y, output))


    def gradient_descent(self, Y, cache, alpha=0.05):
        """performs gradient descent"""

        # getting total number of 'examples'
        m = Y.shape[1]

        # retrieving relevant variables from cache and weights
        W1 = self.__weights['W1']
        W2 = self.__weights['W2']
        W3 = self.__weights['W3']

        b1 = self.__weights['b1']
        b2 = self.__weights['b2']
        b3 = self.__weights['b3']

        X = self.__cache['A0']
        A1 = self.__cache['A1']
        A2 = self.__cache['A2']
        A3 = self.__cache['A3']
        # getting derivatives of layer 3

        dz3 = A3 - Y
        dw3 = np.matmul(dz3, A2.T) / m
        db3 = np.sum(dz3, axis=1, keepdims=True) / m

        # getting derivates of layer 2

        dz2 = (W3.T * dz3) * (A2 * (1 - A2))
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # attempting the derivatives of layer 1
        dz1 = (np.dot(W2.T, dz2) * (A1 * (1 - A1)))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m 

        # update weights dictionary (weights and bias)

        self.__weights['W3'] = W3 - (alpha * dw3)
        self.__weights['b3'] = b3 - (alpha * db3)
        self.__weights['W2'] = W2 - (alpha * dw2)
        self.__weights['b2'] = b2 - (alpha * db2)
        self.__weights['W1'] = W1 - (alpha * dw1)
        self.__weights['b1'] = b1 - (alpha * db1)

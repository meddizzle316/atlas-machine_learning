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
        """this cost function is working but only has up to 7"""
        """or so decimal places the same as the checkers"""
        firstPart = Y * np.log(A)
        secondPart = ((1 - Y) * np.log(1.0000001 - A))
        return np.mean(-(np.mean(firstPart) + (secondPart)))
        # the life of me, I couldn't find the -np.mean bit. -1/m -mean?

    def evaluate(self, X, Y):
        """evaluates the neuron's predictions"""
        """Y: target labels"""
        """X: input data"""
        diff = self.activation(np.dot(self.W, X) + self.__b)
        # I guess prediction just means the dotprod plust bias?
        new_diff = np.where(diff >= 0.5, 1, 0)
        return (new_diff), self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calculates one pass of gradient descent
        on the neuron
        X: input data
        Y: target labels [1, 0] in 1, 0 format
        A: Activated output of neuron
        alpha: learning rate
        updates W and b
        No return
        """

        # getting the number of training examples (to represent m)
        m = X.shape[1]

        # calculating the gradient of the logistic cost function
        # why is the Activation of the neuron important to this function?
        dz = A - Y

        # gradient of the cost function with respect to the biases
        db = np.sum(dz) / m
        # getting the gradient of the cost function with respect to the weights
        # transforming X because dz is 12665 size because of A - Y operation
        # also because X.T is the "training" data 
        dw = np.dot(dz, X.T) / m

        # updating __W and __b
        self.__W = self.W - (alpha * dw)
        self.__b = np.mean(self.b - (alpha * db))

    def activationDerivative(self, Y, A):
        """
        the derivative of the Cross Entropy I think
        Y: target labels
        A: Activated output of neuron
        """
        return (-Y / A) + ((1 - Y) / (1 - A))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A)

        return self.evaluate(X, Y)
